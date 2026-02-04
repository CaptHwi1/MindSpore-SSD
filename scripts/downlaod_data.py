#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ModelArts-compatible data downloader with local fallback.
Handles OBS download via moxing OR direct HTTP for local debugging.
"""
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional

# Try import moxing first (ModelArts environment)
try:
    import moxing as mox
    IS_MODELARTS = True
except (ImportError, ModuleNotFoundError):
    IS_MODELARTS = False
    print("[INFO] Running in LOCAL mode (moxing not available)")

class DataDownloader:
    def __init__(self, 
                 local_root: str = "/cache/dataset",
                 obs_bucket: Optional[str] = None):
        """
        Args:
            local_root: Local dataset root (ModelArts: /cache/dataset, Local: ~/datasets/frugal_factory)
            obs_bucket: OBS bucket path (e.g., "s3://frugal-factory-dataset")
        """
        self.local_root = Path(local_root).resolve()
        self.obs_bucket = obs_bucket
        self.raw_dir = self.local_root / "raw"
        self.coco_dir = self.local_root / "coco"
        self.mindrecord_dir = self.local_root / "mindrecord"
        
        # Create directory structure
        for d in [self.raw_dir, self.coco_dir, self.mindrecord_dir]:
            d.mkdir(parents=True, exist_ok=True)
            print(f"[✓] Created {d}")
    
    def download_from_obs(self, obs_path: str, local_path: str) -> bool:
        """Download from OBS using moxing (ModelArts) or fallback to wget (local)"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        if IS_MODELARTS:
            print(f"[ModelArts] Downloading {obs_path} → {local_path}")
            try:
                mox.file.copy_parallel(obs_path, str(local_path))
                print(f" Download complete: {local_path}")
                return True
            except Exception as e:
                print(f" Moxing download failed: {e}")
                return False
        else:
            # Local fallback: use wget for pre-signed URLs
            print(f"[Local] Downloading via wget: {obs_path} → {local_path}")
            import subprocess
            try:
                cmd = [
                    "wget", "-q", "-O", str(local_path / "temp_download.zip"),
                    obs_path
                ]
                subprocess.run(cmd, check=True, timeout=300)
                print(f"[✓] wget download complete")
                return True
            except Exception as e:
                print(f"[✗] wget download failed: {e}")
                return False
    
    def organize_coco_structure(self):
        """Create COCO directory structure and symlink images"""
        coco_ann_dir = self.coco_dir / "annotations"
        coco_train_dir = self.coco_dir / "train2017"
        coco_val_dir = self.coco_dir / "val2017"
        
        for d in [coco_ann_dir, coco_train_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Symlink val2017 to train2017 for debugging (avoid duplicating data)
        if coco_val_dir.exists():
            if coco_val_dir.is_symlink():
                coco_val_dir.unlink()
            else:
                shutil.rmtree(coco_val_dir)
        coco_val_dir.symlink_to(coco_train_dir, target_is_directory=True)
        print(f"[✓] Created COCO structure with symlink: val2017 → train2017")
    
    def get_paths(self) -> dict:
        """Return standardized paths for config files"""
        return {
            "raw_input": str(self.raw_dir / "input"),
            "raw_annotation": str(self.raw_dir / "annotation"),
            "coco_root": str(self.coco_dir),
            "mindrecord_dir": str(self.mindrecord_dir),
            "annotation_json": str(self.coco_dir / "annotations" / "instances_train2017.json")
        }

def main():
    # Configuration - ADJUST THESE FOR YOUR SETUP
    if IS_MODELARTS:
        # ModelArts paths (fast NVMe storage)
        LOCAL_ROOT = "/cache/dataset"
        OBS_BUCKET = "obs://frugal-factory-dataset"  # Replace with your actual OBS path
    else:
        # Local paths (adjust to your machine)
        LOCAL_ROOT = os.path.expanduser("~/datasets/frugal_factory")
        OBS_BUCKET = None
    
    downloader = DataDownloader(local_root=LOCAL_ROOT, obs_bucket=OBS_BUCKET)
    
    # Step 1: Download raw data (use pre-signed URLs for local testing)
    if IS_MODELARTS:
        # ModelArts: Direct OBS copy (no pre-signed URLs needed)
        downloader.download_from_obs(
            f"{OBS_BUCKET}/input", 
            str(downloader.raw_dir / "input")
        )
        downloader.download_from_obs(
            f"{OBS_BUCKET}/output/dataset-object-detection-0XKL6Fn9SRIDIMPm9DT/annotation/V002",
            str(downloader.raw_dir / "annotation" / "V002")
        )
    else:
        # Local: Use pre-signed URLs (MUST BE FRESH - tokens expire in 24h)
        #  REPLACE THESE WITH CURRENT URLS FROM YOUR DATASET PROVIDER
        INPUT_URL = "https://frugal-factory-dataset.obs.ap-southeast-1.myhuaweicloud.com/input/..."  # Truncated for security
        ANNO_URL = "https://frugal-factory-dataset.obs.ap-southeast-1.myhuaweicloud.com/output/..."  # Truncated for security
        
        downloader.download_from_obs(INPUT_URL, str(downloader.raw_dir / "input_temp"))
        downloader.download_from_obs(ANNO_URL, str(downloader.raw_dir / "annotation_temp"))
    
    # Step 2: Create COCO structure
    downloader.organize_coco_structure()
    
    # Step 3: Output paths for config file
    paths = downloader.get_paths()
    print("\n" + "="*70)
    print(" DATA SETUP COMPLETE")
    print("="*70)
    print(json.dumps(paths, indent=2))
    print("\n NEXT STEP: Update ssd300_config_gpu.yaml with these paths")
    print("="*70)
    
    # Save paths to JSON for config automation
    with open(downloader.local_root / "dataset_paths.json", "w") as f:
        json.dump(paths, f, indent=2)
    print(f"\n Paths saved to: {downloader.local_root / 'dataset_paths.json'}")

if __name__ == "__main__":
    main()