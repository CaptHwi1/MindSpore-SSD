# Save as src/dataset.py - MINDSPORE 1.7.0 COMPATIBLE VERSION
from __future__ import division
import os
import json
import numpy as np
import cv2
import time
from pathlib import Path

import mindspore.dataset as de
# CRITICAL FIX FOR 1.7.0: Use legacy vision API paths
from mindspore.dataset.vision import c_transforms as vision  # ← 1.7.0 compatible
from mindspore.mindrecord import FileWriter
from src.model_utils.config import config
from .box_utils import jaccard_numpy, ssd_bboxes_encode

def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a

def get_image_id_from_filename(filename, id_iter):
    stem = Path(filename).stem
    if stem.isdigit():
        return int(stem)
    return int(hash(stem) & 0xFFFFFFFF)

def random_sample_crop(image, boxes):
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
        
        image_t = image[rect[0]:rect[2], rect[1]:rect[3], :]
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
    cv2.setNumThreads(2)
    
    def _data_aug(image, box, is_training, image_size=(300, 300)):
        ih, iw = image.shape[:2]
        h, w = image_size
        
        if not is_training:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            return img_id, image, np.array([ih, iw], dtype=np.float32)
        
        box = box.astype(np.float32)
        image, box = random_sample_crop(image, box)
        ih, iw = image.shape[:2]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if _rand() < 0.5:
            image = cv2.flip(image, 1)
            if box.size > 0:
                box[:, [1, 3]] = iw - box[:, [3, 1]]
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if box.size > 0:
            box[:, [0, 2]] = box[:, [0, 2]] / ih
            box[:, [1, 3]] = box[:, [1, 3]] / iw
        
        box, label, num_match = ssd_bboxes_encode(box)
        return image, box, label, num_match
    
    return _data_aug(image, box, is_training, image_size=tuple(config.img_shape))

def create_coco_label(is_training=True):
    print(f"[DATASET] Loading COCO annotations (training={is_training})...")
    
    # CRITICAL FIX: Resolve paths for ModelArts 1.7.0
    coco_root = Path(getattr(config, 'coco_root', '/cache/dataset/coco')).expanduser().resolve()
    ann_dir = coco_root / "annotations"
    
    ann_file = getattr(config, 'annotation_file', 'instances_train2017.json') if is_training \
        else getattr(config, 'eval_annotation', 'instances_val2017.json')
    
    ann_path = ann_dir / ann_file
    if not ann_path.exists():
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
            raise FileNotFoundError(f"COCO annotation file not found at {ann_path}")
    
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    cat_id_to_contiguous = {}
    for i, cat in enumerate(coco_data['categories']):
        cat_id_to_contiguous[cat['id']] = i + 1
    
    img_dir_name = 'train2017' if is_training else 'val2017'
    img_dir = coco_root / img_dir_name
    
    if not img_dir.exists():
        fallback_dirs = [coco_root / 'images', coco_root / 'JPEGImages']
        for fd in fallback_dirs:
            if fd.exists():
                img_dir = fd
                print(f"[DATASET] Using fallback image dir: {fd.name}")
                break
        else:
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    filename_to_imgid = {img['file_name']: img['id'] for img in coco_data['images']}
    
    annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        
        if ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
            continue
        
        x, y, w, h = ann['bbox']
        ymin = max(0, y)
        xmin = max(0, x)
        ymax = ymin + h
        xmax = xmin + w
        
        cat_id = ann.get('category_id')
        if cat_id not in cat_id_to_contiguous:
            continue
        
        contiguous_id = cat_id_to_contiguous[cat_id]
        annotations[img_id].append([ymin, xmin, ymax, xmax, contiguous_id])
    
    image_ids = []
    image_path_dict = {}
    annotation_dict = {}
    id_iter = 0
    
    for img in coco_data['images']:
        img_id = img['id']
        filename = img['file_name']
        img_path = img_dir / filename
        
        if not img_path.exists():
            continue
        
        if img_id not in annotations or len(annotations[img_id]) == 0:
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        ih, iw = image.shape[:2]
        fixed_anns = []
        for ann in annotations[img_id]:
            ymin, xmin, ymax, xmax, cat_id = ann
            ymax = min(ymax, ih)
            xmax = min(xmax, iw)
            if ymax <= ymin or xmax <= xmin:
                continue
            fixed_anns.append([ymin, xmin, ymax, xmax, cat_id])
        
        if not fixed_anns:
            continue
        
        deterministic_id = get_image_id_from_filename(filename, id_iter)
        image_ids.append(deterministic_id)
        image_path_dict[deterministic_id] = str(img_path)
        annotation_dict[deterministic_id] = np.array(fixed_anns, dtype=np.float32)
        id_iter += 1
    
    print(f"[DATASET] Loaded {len(image_ids)} valid images from COCO annotations")
    return image_ids, image_path_dict, annotation_dict

def coco_data_to_mindrecord(mindrecord_dir, is_training=True, prefix="coco_train"):
    mindrecord_path = Path(mindrecord_dir) / prefix
    mindrecord_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[MINDRECORD] Creating MindRecord at {mindrecord_path}")
    writer = FileWriter(str(mindrecord_path), shard_num=8)
    
    images, image_path_dict, annotation_dict = create_coco_label(is_training)
    
    if not images:
        raise ValueError("No valid images found for MindRecord creation!")
    
    schema = {
        "img_id": {"type": "int32", "shape": [1]},
        "image": {"type": "bytes"},
        "annotation": {"type": "float32", "shape": [-1, 5]}
    }
    writer.add_schema(schema, "ssd_coco")
    
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

def create_ssd_dataset(mindrecord_file, batch_size=32, device_num=1, rank=0, 
                      is_training=True, num_parallel_workers=4, use_multiprocessing=True):
    if isinstance(mindrecord_file, (list, tuple)):
        check_path = mindrecord_file[0]
    else:
        check_path = str(mindrecord_file).rstrip('0123456789') + '0'
    
    if not Path(check_path).exists():
        raise FileNotFoundError(
            f"MindRecord file not found: {check_path}\n"
            " SOLUTION: Run 'python create_mindrecord.py' first!"
        )
    
    ds = de.MindDataset(
        dataset_files=mindrecord_file,
        columns_list=["img_id", "image", "annotation"],
        num_shards=device_num,
        shard_id=rank,
        shuffle=is_training
    )
    
    # CRITICAL FIX FOR 1.7.0: Use c_transforms.Decode
    decode_op = vision.Decode()  # ← Works in 1.7.0
    ds = ds.map(
        operations=decode_op,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=use_multiprocessing
    )
    
    if is_training:
        output_columns = ["image", "box", "label", "num_match"]
        column_order = ["image", "box", "label", "num_match"]
    else:
        output_columns = ["img_id", "image", "image_shape"]
        column_order = ["img_id", "image", "image_shape"]
    
    ds = ds.map(
        operations=lambda img_id, image, annotation: preprocess_fn(img_id, image, annotation, is_training),
        input_columns=["img_id", "image", "annotation"],
        output_columns=output_columns,
        column_order=column_order,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=use_multiprocessing,
        max_rowsize=16
    )
    
    # CRITICAL FIX FOR 1.7.0: Use c_transforms for all vision ops
    normalize_op = vision.Normalize(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]
    )
    
    if is_training:
        color_adjust_op = vision.RandomColorAdjust(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4
        )
        hwc2chw_op = vision.HWC2CHW()
        
        ds = ds.map(
            operations=[color_adjust_op, normalize_op, hwc2chw_op],
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=use_multiprocessing
        )
    else:
        hwc2chw_op = vision.HWC2CHW()
        ds = ds.map(
            operations=[normalize_op, hwc2chw_op],
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=use_multiprocessing
        )
    
    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.repeat(1)
    
    print(f"[DATASET] Pipeline created:")
    print(f"  - Samples: {ds.get_dataset_size() * batch_size}")
    print(f"  - Batches/epoch: {ds.get_dataset_size()}")
    return ds

def create_mindrecord(dataset="coco", prefix="coco_train", is_training=True):
    mindrecord_dir = Path(getattr(config, 'mindrecord_dir', '/cache/dataset/mindrecord')).expanduser().resolve()
    mindrecord_dir.mkdir(parents=True, exist_ok=True)
    
    mindrecord_path = mindrecord_dir / f"{prefix}0"
    if mindrecord_path.exists():
        print(f"[MINDRECORD] Found existing MindRecord: {mindrecord_path}")
        return str(mindrecord_dir / prefix)
    
    print(f"[MINDRECORD] Creating new MindRecord (not found at {mindrecord_path})")
    
    if dataset.lower() in ["coco", "coco2017"]:
        coco_data_to_mindrecord(mindrecord_dir, is_training, prefix)
    else:
        raise ValueError(f"Unsupported dataset type for 1.7.0: {dataset}. Only 'coco' supported.")
    
    return str(mindrecord_dir / prefix)
