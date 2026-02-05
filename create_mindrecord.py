#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create MindRecord files for SSD training
Supports COCO and VOC formats with automatic validation
"""
import os
import sys
import argparse
import time
from pathlib import Path

# Add repo root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from src.model_utils.config import config
from src.dataset import create_mindrecord as create_mindrecord_impl


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="SSD MindRecord Creator")
    
    # Config path
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='./config/ssd300_config_gpu.yaml',
        help='Path to configuration YAML file'
    )
    
    # Dataset type
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='coco',
        choices=['coco', 'voc'],
        help='Dataset format: coco or voc (default: coco)'
    )
    
    # MindRecord directory override
    parser.add_argument(
        '--mindrecord_dir', 
        type=str, 
        default=None,
        help='Override mindrecord directory from config'
    )
    
    # Prefixes
    parser.add_argument(
        '--prefix_train', 
        type=str, 
        default='coco_train',
        help='Training mindrecord prefix (default: coco_train)'
    )
    parser.add_argument(
        '--prefix_val', 
        type=str, 
        default='coco_val',
        help='Validation mindrecord prefix (default: coco_val)'
    )
    
    # Mode flags
    parser.add_argument(
        '--only_val', 
        action='store_true',
        help='Only create validation mindrecord (skip training)'
    )
    parser.add_argument(
        '--only_train', 
        action='store_true',
        help='Only create training mindrecord (skip validation)'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force recreation even if files exist'
    )
    
    return parser.parse_args()


def check_existing_mindrecords(mindrecord_dir, prefix):
    """Check if mindrecord files already exist"""
    mindrecord_dir = Path(mindrecord_dir)
    pattern = f"{prefix}0.mindrecord"
    return (mindrecord_dir / pattern).exists()


def main():
    args = parse_args()
    
    print("="*70)
    print("MINDRECORD CREATION UTILITY")
    print("="*70)
    print(f"Config file:    {args.config_path}")
    print(f"Dataset type:   {args.dataset.upper()}")
    print(f"Training prefix:{args.prefix_train}")
    print(f"Val prefix:     {args.prefix_val}")
    print("="*70)
    
    # Override config paths if CLI args provided
    if args.mindrecord_dir:
        config.mindrecord_dir = args.mindrecord_dir
        print(f"[INFO] Overriding mindrecord_dir: {config.mindrecord_dir}")
    
    # Resolve absolute paths
    mindrecord_dir = Path(config.mindrecord_dir).expanduser().resolve()
    mindrecord_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] MindRecord output dir: {mindrecord_dir}")
    
    # Check existing files
    train_exists = check_existing_mindrecords(mindrecord_dir, args.prefix_train)
    val_exists = check_existing_mindrecords(mindrecord_dir, args.prefix_val)
    
    if train_exists and val_exists and not args.force:
        print("\n[WARNING] MindRecord files already exist:")
        print(f"  - {mindrecord_dir}/{args.prefix_train}*.mindrecord")
        print(f"  - {mindrecord_dir}/{args.prefix_val}*.mindrecord")
        print("\nUse --force to recreate, or proceed to training.")
        return
    
    # Create training mindrecord
    if not args.only_val:
        if train_exists and not args.force:
            print(f"\n[SKIP] Training mindrecord exists: {args.prefix_train}*.mindrecord")
        else:
            print(f"\n[1/2] Creating TRAINING mindrecord ({args.dataset.upper()} format)...")
            start_time = time.time()
            try:
                train_path = create_mindrecord_impl(
                    dataset=args.dataset,
                    prefix=args.prefix_train,
                    is_training=True
                )
                elapsed = time.time() - start_time
                print(f" Training mindrecord created: {train_path}")
                print(f"   Time: {elapsed:.1f}s")
            except Exception as e:
                print(f"\n FAILED to create training mindrecord: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    
    # Create validation mindrecord
    if not args.only_train:
        if val_exists and not args.force:
            print(f"\n[SKIP] Validation mindrecord exists: {args.prefix_val}*.mindrecord")
        else:
            print(f"\n[2/2] Creating VALIDATION mindrecord ({args.dataset.upper()} format)...")
            start_time = time.time()
            try:
                val_path = create_mindrecord_impl(
                    dataset=args.dataset,
                    prefix=args.prefix_val,
                    is_training=False
                )
                elapsed = time.time() - start_time
                print(f" Validation mindrecord created: {val_path}")
                print(f"   Time: {elapsed:.1f}s")
            except Exception as e:
                print(f"\n FAILED to create validation mindrecord: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    
    # Final summary
    print("\n" + "="*70)
    print("MINDRECORD CREATION COMPLETE")
    print("="*70)
    print(f"Location: {mindrecord_dir}")
    print(f"Files created:")
    print(f"  - {args.prefix_train}0.mindrecord (and 7 shards)")
    print(f"  - {args.prefix_val}0.mindrecord (and 7 shards)")
    print("\nNext steps:")
    print("  1. Validate dataset: python train.py --only_create_dataset True")
    print("  2. Start training:    python train.py --config_path ./config/ssd300_config_gpu.yaml")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
