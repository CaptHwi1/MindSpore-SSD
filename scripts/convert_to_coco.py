#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert custom annotations to COCO format.
Supports: Pascal VOC XML, CSV (labelImg format), Huawei custom JSON.
Auto-detects format and validates output.
"""
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import hashlib
from datetime import datetime

class COCOConverter:
    def __init__(self, 
                 image_dir: str, 
                 annotation_dir: str,
                 output_json: str,
                 categories: Optional[List[Dict]] = None):
        """
        Args:
            image_dir: Path to directory containing images
            annotation_dir: Path to directory containing annotations (XML/JSON/CSV)
            output_json: Path to save COCO JSON
            categories: List of category dicts [{"id": 1, "name": "person"}, ...]
                        If None, auto-detect from annotations
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_json = Path(output_json)
        self.categories = categories or []
        self.category_map = {}  # {name: id}
        self.images = []
        self.annotations = []
        self.annotation_id = 1
        
        assert self.image_dir.exists(), f"Image dir not found: {image_dir}"
        assert self.annotation_dir.exists(), f"Annotation dir not found: {annotation_dir}"
        
        # Auto-detect annotation format
        self.format = self._detect_format()
        print(f"[✓] Detected annotation format: {self.format}")
    
    def _detect_format(self) -> str:
        """Auto-detect annotation format from files in directory"""
        files = list(self.annotation_dir.glob("*"))
        xml_files = [f for f in files if f.suffix.lower() in [".xml", ".XML"]]
        json_files = [f for f in files if f.suffix.lower() in [".json", ".JSON"]]
        csv_files = [f for f in files if f.suffix.lower() in [".csv", ".CSV"]]
        
        if xml_files:
            # Check if Pascal VOC
            sample = xml_files[0]
            try:
                tree = ET.parse(sample)
                root = tree.getroot()
                if root.find("annotation") is not None or root.find("object") is not None:
                    return "pascal_voc"
            except:
                pass
            return "xml_custom"
        
        if json_files:
            # Check if COCO already or Huawei custom
            sample = json_files[0]
            with open(sample) as f:
                data = json.load(f)
                if "images" in data and "annotations" in data:
                    return "coco"  # Already COCO format
                elif "annotations" in data or "objects" in data:
                    return "huawei_custom"
            return "json_custom"
        
        if csv_files:
            return "csv"
        
        raise ValueError(f"Unsupported annotation format in {self.annotation_dir}")
    
    def _get_category_id(self, category_name: str) -> int:
        """Get or create category ID for name"""
        if category_name not in self.category_map:
            cat_id = len(self.category_map) + 1
            self.category_map[category_name] = cat_id
            self.categories.append({"id": cat_id, "name": category_name, "supercategory": "none"})
        return self.category_map[category_name]
    
    def _load_pascal_voc(self, xml_path: Path) -> Tuple[Dict, List[Dict]]:
        """Parse Pascal VOC XML to COCO image + annotations"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Image info
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        # MD5 hash for image ID (deterministic)
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
        
        # Annotations
        anns = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            # COCO format: [x, y, width, height]
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = bbox[2] * bbox[3]
            
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
    
    def _load_huawei_custom(self, json_path: Path) -> Tuple[Dict, List[Dict]]:
        """
        Parse Huawei ModelArts custom annotation format.
        Example structure:
        {
          "annotations": [
            {
              "name": "defect",
              "confidence": 1.0,
              "location": [{"x": 100, "y": 100}, {"x": 200, "y": 200}]
            }
          ],
          "image_name": "img_001.jpg"
        }
        """
        with open(json_path) as f:
            data = json.load(f)
        
        filename = data.get("image_name", data.get("filename", ""))
        img_path = self.image_dir / filename
        
        # Get image dimensions via OpenCV
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
            name = ann.get("name", ann.get("label", "unknown"))
            loc = ann.get("location", [])
            
            if len(loc) >= 2:
                # Bounding box from two corners
                xs = [p["x"] for p in loc]
                ys = [p["y"] for p in loc]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = bbox[2] * bbox[3]
            else:
                # Fallback: skip invalid annotation
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
        """Convert all annotations to COCO format"""
        print(f"[ ] Scanning annotations in {self.annotation_dir}")
        
        if self.format == "pascal_voc":
            loader = self._load_pascal_voc
            pattern = "*.xml"
        elif self.format == "huawei_custom":
            loader = self._load_huawei_custom
            pattern = "*.json"
        else:
            raise NotImplementedError(f"Format '{self.format}' not yet implemented. PRs welcome!")
        
        # Process all annotation files
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
    
    def validate(self) -> bool:
        """Validate COCO structure before saving"""
        errors = []
        
        # Check all images exist
        for img in self.images:
            img_path = self.image_dir / img["file_name"]
            if not img_path.exists():
                errors.append(f"Image missing: {img_path}")
        
        # Check category consistency
        cat_ids = {c["id"] for c in self.categories}
        for ann in self.annotations:
            if ann["category_id"] not in cat_ids:
                errors.append(f"Annotation {ann['id']} has invalid category_id {ann['category_id']}")
        
        if errors:
            print("[✗] Validation errors:")
            for e in errors[:10]:  # Show first 10 errors
                print(f"  - {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
            return False
        
        print("[✓] Validation passed")
        return True
    
    def save(self):
        """Save COCO JSON"""
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
        print(f"    - Images: {len(self.images)}")
        print(f"    - Annotations: {len(self.annotations)}")
        print(f"    - Categories: {len(self.categories)}")

def main():
    #  CONFIGURATION - ADJUST THESE PATHS
    if "MA_JOB_DIR" in os.environ:  # ModelArts detection
        RAW_ROOT = Path("/cache/dataset/raw")
        COCO_ROOT = Path("/cache/dataset/coco")
    else:
        RAW_ROOT = Path.home() / "datasets/frugal_factory/raw"
        COCO_ROOT = Path.home() / "datasets/frugal_factory/coco"
    
    # Paths
    IMAGE_DIR = RAW_ROOT / "input"          # Contains .jpg/.png images
    ANNO_DIR = RAW_ROOT / "annotation/V002" # Contains annotations
    OUTPUT_JSON = COCO_ROOT / "annotations/instances_train2017.json"
    
    # Optional: Pre-define categories (if known)
    CATEGORIES = [
        {"id": 1, "name": "defect_type_1"},
        {"id": 2, "name": "defect_type_2"},
        # Add more as needed
    ]
    
    # Convert
    converter = COCOConverter(
        image_dir=IMAGE_DIR,
        annotation_dir=ANNO_DIR,
        output_json=OUTPUT_JSON,
        categories=CATEGORIES or None
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