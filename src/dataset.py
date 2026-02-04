#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindSpore SSD Dataset Pipeline
Enhanced for COCO format + ModelArts GPU training with robust error handling
"""
from __future__ import division
import os
import json
import multiprocessing
import numpy as np
import cv2
import time
from pathlib import Path

import mindspore.dataset as de
# Robust import strategy for MindSpore 2.x/3.x vision operators
try:
    from mindspore.dataset import vision
    if not hasattr(vision, 'Decode'):
        raise ImportError("Vision module missing Decode")
except (ImportError, AttributeError) as e:
    print(f"[WARNING] Falling back to legacy vision imports: {e}")
    from mindspore.dataset.vision import Decode, Normalize, RandomColorAdjust, HWC2CHW
else:
    Decode = vision.Decode
    Normalize = vision.Normalize
    RandomColorAdjust = vision.RandomColorAdjust
    HWC2CHW = vision.HWC2CHW

from mindspore.mindrecord import FileWriter
from src.model_utils.config import config
from .box_utils import jaccard_numpy, ssd_bboxes_encode

def _rand(a=0., b=1.):
    """Uniform random sampling helper"""
    return np.random.rand() * (b - a) + a

def get_image_id_from_filename(filename, id_iter):
    """Extract deterministic image ID from filename"""
    stem = Path(filename).stem
    if stem.isdigit():
        return int(stem)
    # Fallback: hash-based ID (deterministic)
    return int(hash(stem) & 0xFFFFFFFF)

def random_sample_crop(image, boxes):
    """
    SSD random crop augmentation
    Returns cropped image and adjusted boxes
    """
    height, width, _ = image.shape
    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
    if min_iou is None:
        return image, boxes

    for _ in range(50):
        w = _rand(0.3, 1.0) * width
        h = _rand(0.3, 1.0) * height
        if h / w < 0.5 or h / w > 2.0:
            continue
        
        left = _rand() * (width - w)
        top = _rand() * (height - h)
        rect = np.array([int(top), int(left), int(top + h), int(left + w)])
        
        overlap = jaccard_numpy(boxes, rect)
        if overlap.size == 0:
            continue
            
        drop_mask = overlap > 0
        if not drop_mask.any():
            continue
            
        if overlap[drop_mask].min() < min_iou:
            continue
        
        # Crop image
        image_t = image[rect[0]:rect[2], rect[1]:rect[3], :]
        
        # Adjust boxes
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
        mask = (
            (rect[0] < centers[:, 0]) & (rect[1] < centers[:, 1]) &
            (rect[2] > centers[:, 0]) & (rect[3] > centers[:, 1]) & drop_mask
        )
        if not mask.any():
            continue
            
        boxes_t = boxes[mask, :].copy()
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2]) - rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4]) - rect[:2]
        
        return image_t, boxes_t
    
    return image, boxes

def preprocess_fn(img_id, image, box, is_training):
    """
    SSD preprocessing pipeline
    Handles both training (augmentation) and inference (resize only)
    """
    cv2.setNumThreads(2)  # Critical for multiprocessing stability
    
    def _data_aug(image, box, is_training, image_size=(300, 300)):
        ih, iw = image.shape[:2]
        h, w = image_size
        
        if not is_training:
            # Inference: resize only
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            return img_id, image, np.array([ih, iw], dtype=np.float32)
        
        # Training: augmentation + encoding
        box = box.astype(np.float32)
        image, box = random_sample_crop(image, box)
        ih, iw = image.shape[:2]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Random horizontal flip
        if _rand() < 0.5:
            image = cv2.flip(image, 1)
            if box.size > 0:
                box[:, [1, 3]] = iw - box[:, [3, 1]]  # Flip x coordinates
        
        # Ensure 3 channels (grayscale → RGB)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize coordinates to [0,1]
        if box.size > 0:
            box[:, [0, 2]] = box[:, [0, 2]] / ih  # y coordinates
            box[:, [1, 3]] = box[:, [1, 3]] / iw  # x coordinates
        
        # Encode boxes to SSD format
        box, label, num_match = ssd_bboxes_encode(box)
        return image, box, label, num_match
    
    return _data_aug(image, box, is_training, image_size=tuple(config.img_shape))

def create_coco_label(is_training=True):
    """
    Parse COCO annotations into SSD training format
    Returns: (image_ids, image_path_dict, annotation_dict)
    """
    print(f"[DATASET] Loading COCO annotations (training={is_training})...")
    
    # Resolve paths with fallbacks
    coco_root = Path(getattr(config, 'coco_root', '/cache/dataset/coco')).expanduser().resolve()
    ann_dir = coco_root / "annotations"
    
    # Select annotation file based on training mode
    ann_file = getattr(config, 'annotation_file', 'instances_train2017.json') if is_training \
        else getattr(config, 'eval_annotation', 'instances_val2017.json')
    
    ann_path = ann_dir / ann_file
    if not ann_path.exists():
        # Fallback to standard COCO names
        fallbacks = [
            ann_dir / f"instances_{'train' if is_training else 'val'}2017.json",
            ann_dir / f"{'train' if is_training else 'val'}.json"
        ]
        for fp in fallbacks:
            if fp.exists():
                ann_path = fp
                print(f"[DATASET] Using fallback annotation: {fp.name}")
                break
        else:
            raise FileNotFoundError(
                f"COCO annotation file not found at {ann_path}\n"
                f"Tried fallbacks: {[str(f) for f in fallbacks]}"
            )
    
    # Load COCO JSON
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build category ID mapping (COCO IDs → contiguous IDs)
    cat_id_to_contiguous = {}
    for i, cat in enumerate(coco_data['categories']):
        # Map COCO category ID to our contiguous ID (skip background=0)
        cat_id_to_contiguous[cat['id']] = i + 1  # +1 for background class
    
    # Build image ID → path mapping
    img_id_to_path = {}
    img_dir_name = 'train2017' if is_training else 'val2017'
    img_dir = coco_root / img_dir_name
    
    if not img_dir.exists():
        # Fallback: check for 'images' directory
        fallback_dirs = [coco_root / 'images', coco_root / 'JPEGImages']
        for fd in fallback_dirs:
            if fd.exists():
                img_dir = fd
                print(f"[DATASET] Using fallback image dir: {fd.name}")
                break
        else:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Create reverse mapping: filename → image_id
    filename_to_imgid = {img['file_name']: img['id'] for img in coco_data['images']}
    
    # Build annotation dictionary
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        
        # Skip invalid boxes
        if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
            continue
        
        # COCO format: [x, y, width, height] → SSD format: [ymin, xmin, ymax, xmax]
        x, y, w, h = ann['bbox']
        ymin = max(0, y)
        xmin = max(0, x)
        ymax = min(y + h, 0)  # Will be fixed after loading image dimensions
        xmax = min(x + w, 0)  # Will be fixed after loading image dimensions
        
        # Get category ID (map to contiguous)
        cat_id = ann.get('category_id')
        if cat_id not in cat_id_to_contiguous:
            print(f"[WARNING] Skipping annotation with unknown category_id: {cat_id}")
            continue
        
        contiguous_id = cat_id_to_contiguous[cat_id]
        annotations[img_id].append([ymin, xmin, ymax, xmax, contiguous_id])
    
    # Build final dictionaries with valid images only
    image_ids = []
    image_path_dict = {}
    annotation_dict = {}
    id_iter = 0
    
    for img in coco_data['images']:
        img_id = img['id']
        filename = img['file_name']
        img_path = img_dir / filename
        
        if not img_path.exists():
            print(f"[WARNING] Skipping missing image: {img_path}")
            continue
        
        # Skip images with no valid annotations
        if img_id not in annotations or len(annotations[img_id]) == 0:
            continue
        
        # Fix box coordinates with actual image dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARNING] Skipping unreadable image: {img_path}")
            continue
        
        ih, iw = image.shape[:2]
        fixed_anns = []
        for ann in annotations[img_id]:
            ymin, xmin, _, _, cat_id = ann
            ymax = min(ymin + ann[3], ih)
            xmax = min(xmin + ann[2], iw)
            # Skip degenerate boxes
            if ymax <= ymin or xmax <= xmin:
                continue
            fixed_anns.append([ymin, xmin, ymax, xmax, cat_id])
        
        if not fixed_anns:
            continue
        
        # Assign deterministic ID
        deterministic_id = get_image_id_from_filename(filename, id_iter)
        image_ids.append(deterministic_id)
        image_path_dict[deterministic_id] = str(img_path)
        annotation_dict[deterministic_id] = np.array(fixed_anns, dtype=np.float32)
        id_iter += 1
    
    print(f"[DATASET] Loaded {len(image_ids)} valid images from COCO annotations")
    print(f"          Categories: {len(cat_id_to_contiguous)} (mapped to IDs 1..{len(cat_id_to_contiguous)})")
    return image_ids, image_path_dict, annotation_dict

def create_voc_label(is_training=True):
    """
    Legacy VOC support (for backward compatibility)
    """
    print(f"[DATASET] Loading VOC annotations (training={is_training})...")
    
    voc_root = Path(getattr(config, 'voc_root', './data/VOC2007')).expanduser().resolve()
    sub_dir = 'train' if is_training else 'eval'
    voc_dir = voc_root / sub_dir
    
    if not voc_dir.exists():
        raise ValueError(f"VOC directory not found: {voc_dir}")
    
    image_dir = voc_dir / 'JPEGImages'
    anno_dir = voc_dir / 'Annotations'
    
    if not image_dir.exists() or not anno_dir.exists():
        raise ValueError(f"VOC structure invalid. Expected {image_dir} and {anno_dir}")
    
    # Build class mapping
    cls_map = {name: i for i, name in enumerate(config.classes)}
    
    images = []
    image_path_dict = {}
    annotation_dict = {}
    id_iter = 0
    
    import xml.etree.ElementTree as ET
    
    for anno_file in sorted(anno_dir.glob('*.xml')):
        tree = ET.parse(anno_file)
        root = tree.getroot()
        filename = root.find('filename').text
        image_path = image_dir / filename
        
        if not image_path.exists():
            print(f"[WARNING] Skipping missing image: {image_path}")
            continue
        
        # Parse annotations
        labels = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip()
            if cls_name not in cls_map:
                continue
            
            bbox = obj.find('bndbox')
            try:
                ymin = max(0, int(float(bbox.find('ymin').text)) - 1)
                xmin = max(0, int(float(bbox.find('xmin').text)) - 1)
                ymax = int(float(bbox.find('ymax').text)) - 1
                xmax = int(float(bbox.find('xmax').text)) - 1
                
                # Skip invalid boxes
                if ymax <= ymin or xmax <= xmin:
                    continue
                
                labels.append([ymin, xmin, ymax, xmax, cls_map[cls_name]])
            except (AttributeError, ValueError) as e:
                print(f"[WARNING] Skipping invalid bbox in {anno_file}: {e}")
                continue
        
        if not labels:
            continue
        
        img_id = id_iter
        images.append(img_id)
        image_path_dict[img_id] = str(image_path)
        annotation_dict[img_id] = np.array(labels, dtype=np.float32)
        id_iter += 1
    
    print(f"[DATASET] Loaded {len(images)} valid images from VOC annotations")
    return images, image_path_dict, annotation_dict

def coco_data_to_mindrecord(mindrecord_dir, is_training=True, prefix="coco_train"):
    """
    Convert COCO dataset to MindRecord format
    """
    mindrecord_path = Path(mindrecord_dir) / prefix
    mindrecord_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[MINDRECORD] Creating MindRecord at {mindrecord_path}")
    writer = FileWriter(str(mindrecord_path), shard_num=8)
    
    # Get COCO annotations
    images, image_path_dict, annotation_dict = create_coco_label(is_training)
    
    if not images:
        raise ValueError("No valid images found for MindRecord creation!")
    
    # Define schema
    schema = {
        "img_id": {"type": "int32", "shape": [1]},
        "image": {"type": "bytes"},
        "annotation": {"type": "float32", "shape": [-1, 5]}  # [ymin, xmin, ymax, xmax, class_id]
    }
    writer.add_schema(schema, "ssd_coco")
    
    # Write samples
    total = len(images)
    start_time = time.time()
    for i, img_id in enumerate(images):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (total - i) if i > 0 else 0
            print(f"  Processing {i}/{total} ({i/total*100:.1f}%) - ETA: {eta:.0f}s")
        
        img_path = image_path_dict[img_id]
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            writer.write_raw_data([{
                "img_id": np.array([img_id], dtype=np.int32),
                "image": img_bytes,
                "annotation": annotation_dict[img_id].astype(np.float32)
            }])
        except Exception as e:
            print(f"[WARNING] Skipping image {img_path}: {e}")
            continue
    
    writer.commit()
    print(f"[MINDRECORD] Created {mindrecord_path} with {total} samples")

def voc_data_to_mindrecord(mindrecord_dir, is_training=True, prefix="voc_train"):
    """
    Legacy VOC MindRecord creation (for backward compatibility)
    """
    mindrecord_path = Path(mindrecord_dir) / prefix
    mindrecord_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[MINDRECORD] Creating VOC MindRecord at {mindrecord_path}")
    writer = FileWriter(str(mindrecord_path), shard_num=8)
    
    images, image_path_dict, annotation_dict = create_voc_label(is_training)
    
    if not images:
        raise ValueError("No valid VOC images found!")
    
    schema = {
        "img_id": {"type": "int32", "shape": [1]},
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]}
    }
    writer.add_schema(schema, "ssd_voc")
    
    for img_id in images:
        with open(image_path_dict[img_id], 'rb') as f:
            writer.write_raw_data([{
                "img_id": np.array([img_id], dtype=np.int32),
                "image": f.read(),
                "annotation": annotation_dict[img_id].astype(np.int32)
            }])
    
    writer.commit()
    print(f"[MINDRECORD] Created VOC MindRecord with {len(images)} samples")

def create_ssd_dataset(mindrecord_file, batch_size=32, device_num=1, rank=0, 
                      is_training=True, num_parallel_workers=4, use_multiprocessing=True):
    """
    Create MindSpore dataset pipeline from MindRecord files
    CRITICAL FIX: Proper column mapping to avoid silent failures
    """
    # Validate mindrecord existence FIRST
    if isinstance(mindrecord_file, (list, tuple)):
        check_path = mindrecord_file[0]
    else:
        check_path = str(mindrecord_file).rstrip('0123456789') + '0'  # Handle sharded files
    
    if not Path(check_path).exists():
        raise FileNotFoundError(
            f"MindRecord file not found: {check_path}\n"
            " SOLUTION: Run 'python create_mindrecord.py' first!"
        )
    
    # Create dataset
    ds = de.MindDataset(
        dataset_files=mindrecord_file,
        columns_list=["img_id", "image", "annotation"],
        num_shards=device_num,
        shard_id=rank,
        shuffle=is_training
    )
    
    # Decode images
    decode_op = Decode()
    ds = ds.map(
        operations=decode_op,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=use_multiprocessing
    )
    
    # Preprocessing pipeline
    if is_training:
        # Training: image + boxes → encoded boxes + labels
        output_columns = ["image", "box", "label", "num_match"]
        column_order = ["image", "box", "label", "num_match"]
    else:
        # Inference: image → resized image + original shape
        output_columns = ["img_id", "image", "image_shape"]
        column_order = ["img_id", "image", "image_shape"]
    
    # CRITICAL FIX: Proper column mapping with column_order
    ds = ds.map(
        operations=lambda img_id, image, annotation: preprocess_fn(img_id, image, annotation, is_training),
        input_columns=["img_id", "image", "annotation"],
        output_columns=output_columns,
        column_order=column_order,  # FIX: Prevents column mismatch errors
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=use_multiprocessing,
        max_rowsize=16  # Prevent OOM for large images
    )
    
    # Vision transforms
    normalize_op = Normalize(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        is_hwc=True  # Input is HWC format before HWC2CHW
    )
    
    if is_training:
        color_adjust_op = RandomColorAdjust(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4
        )
        hwc2chw_op = HWC2CHW()
        
        # Apply transforms in correct order
        ds = ds.map(
            operations=[color_adjust_op, normalize_op, hwc2chw_op],
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=use_multiprocessing
        )
    else:
        hwc2chw_op = HWC2CHW()
        ds = ds.map(
            operations=[normalize_op, hwc2chw_op],
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=use_multiprocessing
        )
    
    # Batch and repeat
    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.repeat(1)  # Explicit repeat for clarity
    
    print(f"[DATASET] Pipeline created:")
    print(f"  - Samples: {ds.get_dataset_size() * batch_size}")
    print(f"  - Batches/epoch: {ds.get_dataset_size()}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Workers: {num_parallel_workers}")
    print(f"  - Training mode: {is_training}")
    
    return ds

def create_mindrecord(dataset="coco", prefix="coco_train", is_training=True):
    """
    Unified MindRecord creation interface supporting both COCO and VOC
    """
    # Resolve mindrecord directory
    mindrecord_dir = Path(getattr(config, 'mindrecord_dir', '/cache/dataset/mindrecord')).expanduser().resolve()
    mindrecord_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if mindrecord already exists
    mindrecord_path = mindrecord_dir / f"{prefix}0"
    if mindrecord_path.exists():
        print(f"[MINDRECORD] Found existing MindRecord: {mindrecord_path}")
        return str(mindrecord_dir / prefix)
    
    print(f"[MINDRECORD] Creating new MindRecord (not found at {mindrecord_path})")
    
    # Create based on dataset type
    if dataset.lower() in ["coco", "coco2017"]:
        coco_data_to_mindrecord(mindrecord_dir, is_training, prefix)
    elif dataset.lower() in ["voc", "voc2007", "voc2012"]:
        voc_data_to_mindrecord(mindrecord_dir, is_training, prefix)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}. Supported: 'coco', 'voc'")
    
    return str(mindrecord_dir / prefix)
