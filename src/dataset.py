from __future__ import division
import os
import json
import multiprocessing
import xml.etree.ElementTree as et
import numpy as np
import cv2

import mindspore.dataset as de
# Robust import strategy for MindSpore 2.x/3.x vision operators
try:
    from mindspore.dataset import vision
    if not hasattr(vision, 'Decode'):
        raise ImportError
except (ImportError, AttributeError):
    import mindspore.dataset.vision.c_transforms as vision

from mindspore.mindrecord import FileWriter
from src.model_utils.config import config
from .box_utils import jaccard_numpy, ssd_bboxes_encode

def _rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a

def get_imageId_from_fileName(filename, id_iter):
    filename = os.path.splitext(filename)[0]
    if filename.isdigit():
        return int(filename)
    return id_iter

def random_sample_crop(image, boxes):
    height, width, _ = image.shape
    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
    if min_iou is None:
        return image, boxes

    for _ in range(50):
        w, h = _rand(0.3, 1.0) * width, _rand(0.3, 1.0) * height
        if h / w < 0.5 or h / w > 2: continue
        left, top = _rand() * (width - w), _rand() * (height - h)
        rect = np.array([int(top), int(left), int(top + h), int(left + w)])
        overlap = jaccard_numpy(boxes, rect)
        drop_mask = overlap > 0
        if not drop_mask.any() or (overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2)):
            continue
        image_t = image[rect[0]:rect[2], rect[1]:rect[3], :]
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
        mask = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1]) * (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1]) * drop_mask
        if not mask.any(): continue
        boxes_t = boxes[mask, :].copy()
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2]) - rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4]) - rect[:2]
        return image_t, boxes_t
    return image, boxes

def preprocess_fn(img_id, image, box, is_training):
    cv2.setNumThreads(2)
    def _data_aug(image, box, is_training, image_size=(300, 300)):
        ih, iw, _ = image.shape
        h, w = image_size
        if not is_training:
            image = cv2.resize(image, (w, h))
            return img_id, image, np.array((ih, iw), np.float32)
        box = box.astype(np.float32)
        image, box = random_sample_crop(image, box)
        ih, iw, _ = image.shape
        image = cv2.resize(image, (w, h))
        flip = _rand() < .5
        if flip: image = cv2.flip(image, 1)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)
        box[:, [0, 2]], box[:, [1, 3]] = box[:, [0, 2]] / ih, box[:, [1, 3]] / iw
        if flip: box[:, [1, 3]] = 1 - box[:, [3, 1]]
        box, label, num_match = ssd_bboxes_encode(box)
        return image, box, label, num_match
    return _data_aug(image, box, is_training, image_size=config.img_shape)

def create_voc_label(is_training):
    voc_root = config.voc_root
    cls_map = {name: i for i, name in enumerate(config.classes)}
    sub_dir = 'train' if is_training else 'eval'
    voc_dir = os.path.join(voc_root, sub_dir)
    if not os.path.isdir(voc_dir): raise ValueError(f'Path {voc_dir} not found.')
    image_dir, anno_dir = os.path.join(voc_dir, 'JPEGImages'), os.path.join(voc_dir, 'Annotations')
    
    images, image_files_dict, image_anno_dict = [], {}, {}
    id_iter = 0
    for anno_file in os.listdir(anno_dir):
        if not anno_file.endswith('xml'): continue
        tree = et.parse(os.path.join(anno_dir, anno_file))
        root = tree.getroot()
        file_name = root.find('filename').text
        image_path = os.path.join(image_dir, file_name)
        if not os.path.isfile(image_path): continue
        labels = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in cls_map: continue
            bbox = obj.find('bndbox')
            labels.append([int(float(bbox.find('ymin').text))-1, int(float(bbox.find('xmin').text))-1,
                           int(float(bbox.find('ymax').text))-1, int(float(bbox.find('xmax').text))-1, cls_map[cls_name]])
        if labels:
            img_id = id_iter
            images.append(img_id)
            image_files_dict[img_id], image_anno_dict[img_id] = image_path, np.array(labels)
            id_iter += 1
    return images, image_files_dict, image_anno_dict

def voc_data_to_mindrecord(mindrecord_dir, is_training, prefix="ssd.mindrecord"):
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, 8)
    images, image_path_dict, image_anno_dict = create_voc_label(is_training)
    schema = {"img_id": {"type": "int32", "shape": [1]}, "image": {"type": "bytes"}, "annotation": {"type": "int32", "shape": [-1, 5]}}
    writer.add_schema(schema, "ssd_json")
    for img_id in images:
        with open(image_path_dict[img_id], 'rb') as f:
            writer.write_raw_data([{"img_id": np.array([img_id], dtype=np.int32), "image": f.read(), "annotation": np.array(image_anno_dict[img_id], dtype=np.int32)}])
    writer.commit()

def create_ssd_dataset(mindrecord_file, batch_size=32, device_num=1, rank=0, is_training=True, 
                       num_parallel_workers=4, use_multiprocessing=True):
    ds = de.MindDataset(mindrecord_file, columns_list=["img_id", "image", "annotation"], num_shards=device_num, shard_id=rank)
    
    decode_op = vision.Decode()
    normalize_op = vision.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    color_adjust_op = vision.RandomColorAdjust(0.4, 0.4, 0.4)
    hwc2chw_op = vision.HWC2CHW()

    # Apply Decode
    ds = ds.map(operations=decode_op, input_columns=["image"])
    
    # Preprocess Logic
    compose_func = (lambda img_id, image, annotation: preprocess_fn(img_id, image, annotation, is_training))
    
    if is_training:
        output_cols = ["image", "box", "label", "num_match"]
        trans = [color_adjust_op, normalize_op, hwc2chw_op]
    else:
        output_cols = ["img_id", "image", "image_shape"]
        trans = [normalize_op, hwc2chw_op]

    # FIX: Added column_order to handle mismatched input/output column lengths
    ds = ds.map(operations=compose_func, 
                input_columns=["img_id", "image", "annotation"], 
                output_columns=output_cols, 
                column_order=output_cols,
                num_parallel_workers=num_parallel_workers,
                python_multiprocessing=use_multiprocessing)

    ds = ds.map(operations=trans, 
                input_columns=["image"], 
                num_parallel_workers=num_parallel_workers,
                python_multiprocessing=use_multiprocessing)
    
    return ds.batch(batch_size, drop_remainder=True)

def create_mindrecord(dataset="voc", prefix="ssd.mindrecord", is_training=True):
    mindrecord_dir = os.path.join(config.data_path, config.mindrecord_dir)
    if not os.path.exists(os.path.join(mindrecord_dir, prefix + "0")):
        if not os.path.exists(mindrecord_dir): os.makedirs(mindrecord_dir)
        voc_data_to_mindrecord(mindrecord_dir, is_training, prefix)
    return os.path.join(mindrecord_dir, prefix + "0")
