# Save as create_mindrecord.py at repo root
#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_utils.config import config
from src.dataset import create_mindrecord

def main():
    print("="*70)
    print("Creating COCO MindRecord files for MindSpore 1.7.0")
    print("="*70)
    
    # Force COCO paths for 1.7.0
    config.coco_root = "/cache/dataset/coco"
    config.mindrecord_dir = "/cache/dataset/mindrecord"
    
    print(f"Training mindrecord...")
    create_mindrecord(dataset="coco", prefix="coco_train", is_training=True)
    
    print(f"\nValidation mindrecord...")
    create_mindrecord(dataset="coco", prefix="coco_val", is_training=False)
    
    print("\n MindRecord creation complete!")
    print(f"   Location: {config.mindrecord_dir}")
    print(f"   Files: coco_train0.mindrecord, coco_val0.mindrecord")

if __name__ == "__main__":
    main()
