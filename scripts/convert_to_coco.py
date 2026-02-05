# Save as scripts/convert_to_coco.py - MODELARTS 1.7.0 COMPATIBLE
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert custom annotations to COCO format (ModelArts 1.7.0 compatible)
"""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import cv2

class COCOConverter:
    def __init__(self, image_dir, annotation_dir, output_json, categories=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_json = Path(output_json)
        self.categories = categories or []
        self.category_map = {}
        self.images = []
        self.annotations = []
        self.annotation_id = 1
        
        # CRITICAL FIX FOR MODELARTS: Resolve to /cache paths
        if not self.image_dir.exists() and 'MA_JOB_DIR' in os.environ:
            # Try ModelArts cache paths
            candidates = [
                Path("/cache/dataset/raw/input"),
                Path("/cache/dataset/coco/train2017"),
                Path("/home/ma-user/work/data/VOC2007/JPEGImages")
            ]
            for cand in candidates:
                if cand.exists():
                    self.image_dir = cand
                    print(f"[AUTO-RESOLVE] Using image dir: {cand}")
                    break
        
        assert self.image_dir.exists(), f"Image dir not found: {image_dir} (tried auto-resolve)"
        assert self.annotation_dir.exists(), f"Annotation dir not found: {annotation_dir}"
        
        self.format = self._detect_format()
        print(f"[✓] Detected annotation format: {self.format}")
    
    def _detect_format(self):
        files = list(self.annotation_dir.glob("*"))
        json_files = [f for f in files if f.suffix.lower() in [".json", ".JSON"]]
        
        if json_files:
            sample = json_files[0]
            with open(sample) as f:
                data = json.load(f)
                if "annotations" in data or "objects" in data:
                    return "huawei_custom"
            return "json_custom"
        
        raise ValueError(f"Unsupported annotation format in {self.annotation_dir} (only JSON supported for 1.7.0)")
    
    def _get_category_id(self, category_name):
        if category_name not in self.category_map:
            cat_id = len(self.category_map) + 1
            self.category_map[category_name] = cat_id
            self.categories.append({"id": cat_id, "name": category_name, "supercategory": "none"})
        return self.category_map[category_name]
    
    def _load_huawei_custom(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        
        filename = data.get("image_name", data.get("filename", ""))
        img_path = self.image_dir / filename
        
        if not img_path.exists():
            # Try common filename variations
            variations = [filename, Path(filename).name, f"{Path(filename).stem}.jpg", f"{Path(filename).stem}.png"]
            for var in variations:
                alt_path = self.image_dir / var
                if alt_path.exists():
                    img_path = alt_path
                    filename = var
                    break
        
        if img_path.exists():
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
        else:
            print(f"[!] Warning: Image not found: {img_path}. Using placeholder dims.")
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
        
        anns = []
        annotations = data.get("annotations", data.get("objects", []))
        for ann in annotations:
            name = ann.get("name", ann.get("label", "defect"))
            loc = ann.get("location", [])
            
            if len(loc) >= 2:
                xs = [p["x"] for p in loc]
                ys = [p["y"] for p in loc]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = bbox[2] * bbox[3]
            else:
                continue
            
            anns.append({
                "id": self.annotation_id,
                "image_id": img_id,
                "category_id": self._get_category_id(name),
                "bbox": bbox,
                "area": area,
                "segmentation": [],
                "iscrowd": 0
            })
            self.annotation_id += 1
        
        return image, anns
    
    def convert(self):
        print(f"[ ] Scanning annotations in {self.annotation_dir}")
        
        loader = self._load_huawei_custom
        pattern = "*.json"
        
        ann_files = list(self.annotation_dir.glob(pattern))
        print(f"[ ] Found {len(ann_files)} annotation files")
        
        for i, ann_file in enumerate(ann_files, 1):
            try:
                image, anns = loader(ann_file)
                self.images.append(image)
                self.annotations.extend(anns)
                
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(ann_files)} files...")
            except Exception as e:
                print(f"[!] Skipping {ann_file.name}: {e}")
        
        print(f"[✓] Converted {len(self.images)} images, {len(self.annotations)} annotations")
        print(f"[✓] Detected {len(self.categories)} categories: {[c['name'] for c in self.categories]}")
    
    def validate(self):
        errors = []
        
        for img in self.images:
            img_path = self.image_dir / img["file_name"]
            if not img_path.exists():
                errors.append(f"Image missing: {img_path}")
        
        cat_ids = {c["id"] for c in self.categories}
        for ann in self.annotations:
            if ann["category_id"] not in cat_ids:
                errors.append(f"Annotation {ann['id']} has invalid category_id {ann['category_id']}")
        
        if errors:
            print("[✗] Validation errors:")
            for e in errors[:10]:
                print(f"  - {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
            return False
        
        print("[✓] Validation passed")
        return True
    
    def save(self):
        coco_json = {
            "info": {
                "description": "Converted dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Auto-converted",
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
        
        print(f"[✓] Saved COCO JSON to: {self.output_json}")

def main():
    # CRITICAL FIX: Auto-resolve paths for ModelArts 1.7.0
    if 'MA_JOB_DIR' in os.environ:
        RAW_ROOT = Path("/cache/dataset/raw")
        COCO_ROOT = Path("/cache/dataset/coco")
    else:
        RAW_ROOT = Path.home() / "datasets/frugal_factory/raw"
        COCO_ROOT = Path.home() / "datasets/frugal_factory/coco"
    
    # Handle symlinked val2017 → train2017
    IMAGE_DIR = RAW_ROOT / "input"
    if not IMAGE_DIR.exists():
        IMAGE_DIR = COCO_ROOT / "train2017"
    
    ANNO_DIR = RAW_ROOT / "annotation" / "V002"
    if not ANNO_DIR.exists():
        ANNO_DIR = RAW_ROOT / "annotation"
    
    OUTPUT_JSON = COCO_ROOT / "annotations/instances_train2017.json"
    
    CATEGORIES = [
        {"id": 1, "name": "Normal"},
        {"id": 2, "name": "Defective"}
    ]
    
    converter = COCOConverter(
        image_dir=IMAGE_DIR,
        annotation_dir=ANNO_DIR,
        output_json=OUTPUT_JSON,
        categories=CATEGORIES
    )
    
    converter.convert()
    
    if converter.validate():
        converter.save()
        print("\n Conversion successful! Next steps:")
        print("  1. Run create_mindrecord.py to generate .mindrecord files")
        print("  2. Update ssd300_config_gpu.yaml with correct paths")
        print("  3. Train with: python train.py --config_path ./config/ssd300_config_gpu.yaml")
    else:
        print("\n[✗] Conversion failed validation. Fix errors and retry.")
        exit(1)

if __name__ == "__main__":
    main()
