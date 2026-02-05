#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Super-robust COCO converter for ModelArts + MindSpore 1.7.0
Handles all path variations with explicit debug output
"""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import cv2
import sys

def resolve_modelarts_paths():
    """Auto-resolve paths for ModelArts environment with exhaustive fallbacks"""
    is_modelarts = 'MA_JOB_DIR' in os.environ or 'DLS_NOTEBOOK_BASE_URL' in os.environ
    
    print("="*70)
    print("MODELARTS PATH RESOLUTION")
    print("="*70)
    print(f"Environment detected: {'ModelArts' if is_modelarts else 'Local'}")
    
    # Define candidate paths IN PRIORITY ORDER
    candidates = {
        'raw_input': [
            Path("/cache/dataset/raw/input"),
            Path("/cache/dataset/raw/images"),
            Path("/cache/dataset/coco/train2017"),
            Path("/home/ma-user/work/data/VOC2007/JPEGImages"),
            Path("/home/ma-user/datasets/frugal_factory/raw/input"),
        ],
        'raw_annotation': [
            Path("/cache/dataset/raw/annotation/V002"),
            Path("/cache/dataset/raw/annotation"),
            Path("/cache/dataset/raw/annotations"),
            Path("/home/ma-user/work/data/VOC2007/Annotations"),
            Path("/home/ma-user/datasets/frugal_factory/raw/annotation/V002"),
        ],
        'coco_root': [
            Path("/cache/dataset/coco"),
            Path("/cache/dataset/coco2017"),
            Path("/home/ma-user/work/coco_ori"),
            Path("/home/ma-user/datasets/frugal_factory/coco"),
        ]
    }
    
    resolved = {}
    for key, paths in candidates.items():
        for p in paths:
            if p.exists():
                resolved[key] = p.resolve()
                print(f" {key:20s} → {p} (exists)")
                break
        else:
            # Create if it's the primary ModelArts path
            if is_modelarts and key in ['coco_root', 'raw_input']:
                primary = paths[0]
                primary.mkdir(parents=True, exist_ok=True)
                resolved[key] = primary.resolve()
                print(f" {key:20s} → {primary} (created)")
            else:
                resolved[key] = paths[0].resolve()
                print(f"  {key:20s} → {paths[0]} (missing - will attempt anyway)")
    
    # Ensure COCO structure
    coco_root = resolved['coco_root']
    (coco_root / "annotations").mkdir(parents=True, exist_ok=True)
    (coco_root / "train2017").mkdir(parents=True, exist_ok=True)
    
    # Create val2017 symlink if missing
    val_dir = coco_root / "val2017"
    if not val_dir.exists():
        try:
            val_dir.symlink_to(coco_root / "train2017", target_is_directory=True)
            print(f" Created symlink: val2017 → train2017")
        except Exception as e:
            print(f"  Failed to create symlink: {e}")
    
    print("="*70)
    return resolved

class COCOConverter:
    def __init__(self, image_dir, annotation_dir, output_json, categories=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_json = Path(output_json)
        self.categories = categories or [
            {"id": 1, "name": "Normal", "supercategory": "defect"},
            {"id": 2, "name": "Defective", "supercategory": "defect"}
        ]
        self.category_map = {cat['name']: cat['id'] for cat in self.categories}
        self.images = []
        self.annotations = []
        self.annotation_id = 1
        
        print(f"\n[INIT] Image dir: {self.image_dir}")
        print(f"[INIT] Annotation dir: {self.annotation_dir}")
        print(f"[INIT] Output JSON: {self.output_json}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")
        
        self.format = self._detect_format()
        print(f"[INIT] Detected format: {self.format}")
    
    def _detect_format(self):
        json_files = list(self.annotation_dir.glob("*.json"))
        if json_files:
            sample = json_files[0]
            try:
                with open(sample) as f:
                    data = json.load(f)
                if "annotations" in data or "objects" in data or "image_name" in data:
                    return "huawei_custom"
            except:
                pass
            return "coco_json"
        raise ValueError(f"No JSON annotations found in {self.annotation_dir}")
    
    def _load_huawei_custom(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        
        # Handle multiple possible field names for image filename
        filename_candidates = [
            data.get("image_name"),
            data.get("filename"),
            data.get("file_name"),
            Path(json_path).stem + ".jpg",
            Path(json_path).stem + ".png"
        ]
        filename = next((f for f in filename_candidates if f), None)
        
        if not filename:
            raise ValueError(f"Could not determine image filename from {json_path}")
        
        # Try multiple image path variations
        img_variations = [
            self.image_dir / filename,
            self.image_dir / Path(filename).name,
            self.image_dir / f"{Path(filename).stem}.jpg",
            self.image_dir / f"{Path(filename).stem}.png",
        ]
        
        img_path = None
        for var in img_variations:
            if var.exists():
                img_path = var
                filename = var.name
                break
        
        if not img_path or not img_path.exists():
            print(f"  Image not found for annotation {json_path.name}. Skipping.")
            return None, None
        
        # Read image dimensions
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("OpenCV could not read image")
            height, width = img.shape[:2]
        except Exception as e:
            print(f"  Failed to read image {img_path}: {e}. Using defaults.")
            width, height = 640, 480
        
        img_id = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)
        
        image = {
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Extract annotations
        anns = []
        annotations = data.get("annotations", data.get("objects", []))
        if isinstance(annotations, dict):  # Handle single annotation case
            annotations = [annotations]
        
        for ann in annotations:
            name = ann.get("name", ann.get("label", "Defective"))
            if name not in self.category_map:
                name = "Defective"  # Default fallback
            
            # Handle different location formats
            loc = ann.get("location", [])
            if not loc or len(loc) < 2:
                continue
            
            try:
                xs = [float(p["x"]) for p in loc]
                ys = [float(p["y"]) for p in loc]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = bbox[2] * bbox[3]
                if area <= 0:
                    continue
            except (KeyError, ValueError, TypeError) as e:
                print(f"  Skipping invalid bbox in {json_path.name}: {e}")
                continue
            
            anns.append({
                "id": self.annotation_id,
                "image_id": img_id,
                "category_id": self.category_map[name],
                "bbox": bbox,
                "area": area,
                "segmentation": [],
                "iscrowd": 0
            })
            self.annotation_id += 1
        
        return image, anns
    
    def convert(self):
        print(f"\n[CONVERT] Scanning {self.annotation_dir} for annotations...")
        json_files = list(self.annotation_dir.glob("*.json"))
        print(f"[CONVERT] Found {len(json_files)} JSON annotation files")
        
        valid_count = 0
        for i, ann_file in enumerate(sorted(json_files), 1):
            try:
                image, anns = self._load_huawei_custom(ann_file)
                if image is None or not anns:
                    continue
                
                self.images.append(image)
                self.annotations.extend(anns)
                valid_count += 1
                
                if i % 100 == 0 or i == len(json_files):
                    print(f"  Processed {i}/{len(json_files)} files ({valid_count} valid)")
            except Exception as e:
                print(f"  Error processing {ann_file.name}: {e}")
                continue
        
        print(f"\n[CONVERT] Summary:")
        print(f"  Total annotation files: {len(json_files)}")
        print(f"  Valid images with annotations: {len(self.images)}")
        print(f"  Total bounding boxes: {len(self.annotations)}")
        print(f"  Categories: {list(self.category_map.keys())}")
    
    def save(self):
        if not self.images:
            raise ValueError("No valid images to save! Check annotation format.")
        
        coco_json = {
            "info": {
                "description": "Frugal Factory Defect Detection Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "ModelArts Auto-Converter",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 0, "name": "Unknown", "url": ""}],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w") as f:
            json.dump(coco_json, f, indent=2)
        
        print(f"\n COCO JSON saved to: {self.output_json}")
        print(f"   - Images: {len(self.images)}")
        print(f"   - Annotations: {len(self.annotations)}")
        print(f"   - File size: {self.output_json.stat().st_size / 1024:.1f} KB")

def main():
    # STEP 1: Resolve paths with exhaustive ModelArts detection
    paths = resolve_modelarts_paths()
    
    # STEP 2: Symlink images to COCO structure if needed
    coco_root = paths['coco_root']
    train_dir = coco_root / "train2017"
    
    # Only copy/symlink if train2017 is empty
    if not any(train_dir.iterdir()):
        print("\n[IMAGE SETUP] Populating COCO image directory...")
        image_sources = [
            paths['raw_input'],
            paths['raw_input'].parent / "images",
            Path("/cache/dataset/raw/JPEGImages")
        ]
        
        for src in image_sources:
            if src.exists():
                print(f"  Symlinking images from {src} → {train_dir}")
                for img in src.glob("*.[jJ][pP][gG]"):
                    (train_dir / img.name).symlink_to(img.resolve(), target_is_directory=False)
                for img in src.glob("*.[pP][nN][gG]"):
                    (train_dir / img.name).symlink_to(img.resolve(), target_is_directory=False)
                break
        else:
            print("  No images found to symlink. Training will fail unless images exist in train2017/")
    
    # STEP 3: Convert annotations
    print("\n" + "="*70)
    print("ANNOTATION CONVERSION")
    print("="*70)
    
    converter = COCOConverter(
        image_dir=train_dir,
        annotation_dir=paths['raw_annotation'],
        output_json=coco_root / "annotations/instances_train2017.json",
        categories=None  # Use default Normal/Defective
    )
    
    converter.convert()
    converter.save()
    
    # STEP 4: Create minimal val2017 annotation (copy of train for validation)
    val_json = coco_root / "annotations/instances_val2017.json"
    if not val_json.exists():
        print(f"\n[VAL SETUP] Creating minimal validation annotation (copy of train)...")
        with open(coco_root / "annotations/instances_train2017.json") as f:
            train_data = json.load(f)
        
        # Take first 10% of images for validation
        split_idx = max(1, len(train_data['images']) // 10)
        val_data = {
            "info": train_data['info'],
            "licenses": train_data['licenses'],
            "images": train_data['images'][:split_idx],
            "annotations": [a for a in train_data['annotations'] 
                          if a['image_id'] in [img['id'] for img in train_data['images'][:split_idx]]],
            "categories": train_data['categories']
        }
        
        with open(val_json, 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f" Created validation annotation: {val_json}")
    
    print("\n" + "="*70)
    print(" CONVERSION COMPLETE - Ready for MindRecord creation")
    print("="*70)
    print(f"COCO Root: {coco_root}")
    print(f"Train annotations: {coco_root / 'annotations/instances_train2017.json'}")
    print(f"Val annotations:   {coco_root / 'annotations/instances_val2017.json'}")
    print(f"Train images:      {train_dir} ({len(list(train_dir.glob('*.jpg'))) + len(list(train_dir.glob('*.png')))})")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
