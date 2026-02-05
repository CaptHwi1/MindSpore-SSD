#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Moxing Adapter for ModelArts
Enhanced for COCO workflow with safe attribute access and NVMe-optimized paths
"""
import os
import functools
import time
import mindspore as ms
from src.model_utils.config import config

_global_sync_count = 0

def get_device_id():
    """Get device ID from environment (ModelArts standard)"""
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

def get_device_num():
    """Get total device count (for distributed training)"""
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)

def get_rank_id():
    """Get global rank ID (for distributed training)"""
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)

def get_job_id():
    """Get ModelArts job ID for logging"""
    job_id = os.getenv('JOB_ID', 'default')
    return job_id if job_id != "" else "default"

def sync_data(from_path, to_path, is_upload=False):
    """
    Synchronize data between OBS and local storage with device coordination.
    
    Args:
        from_path: Source path (OBS URL or local path)
        to_path: Destination path (local path or OBS URL)
        is_upload: True if uploading to OBS (skips lock for faster upload)
    """
    # Skip sync for empty/invalid paths
    if not from_path or not to_path or from_path == to_path:
        print(f"[MOXING] Skipping sync: invalid paths ({from_path} → {to_path})")
        return
    
    # Early exit if source doesn't exist (for uploads)
    if not is_upload and not (from_path.startswith(('s3://', 'obs://')) or os.path.exists(from_path)):
        print(f"[MOXING] Skipping sync: source not found ({from_path})")
        return
    
    try:
        import moxing as mox
    except ImportError:
        print(f"[MOXING] Moxing not available (local mode). Skipping sync: {from_path} → {to_path}")
        return
    
    global _global_sync_count
    sync_lock = f"/tmp/copy_sync.lock.{_global_sync_count}.{get_device_id()}"
    _global_sync_count += 1
    
    # Only device 0 performs download (prevents 8x redundant downloads on 8-GPU node)
    should_sync = (get_device_id() % min(get_device_num(), 8) == 0)
    
    if should_sync and not os.path.exists(sync_lock):
        print(f"[MOXING] {'UPLOAD' if is_upload else 'DOWNLOAD'}: {from_path} → {to_path}")
        
        # Ensure destination directory exists (critical fix)
        if not is_upload and not from_path.startswith(('s3://', 'obs://')):
            os.makedirs(to_path, exist_ok=True)
        
        try:
            start_time = time.time()
            mox.file.copy_parallel(from_path, to_path)
            elapsed = time.time() - start_time
            print(f"[MOXING]  Sync completed in {elapsed:.1f}s ({from_path} → {to_path})")
            
            # Create lock file to signal completion
            with open(sync_lock, 'w') as f:
                f.write(f"Completed at {time.time()}")
        except Exception as e:
            print(f"[MOXING]  Sync FAILED: {from_path} → {to_path}")
            print(f"         Error: {type(e).__name__}: {e}")
            # Don't create lock on failure – allows retry
    
    # Wait for device 0 to complete sync (other devices block here)
    if not should_sync:
        print(f"[MOXING] Device {get_device_id()}: Waiting for sync completion...")
        wait_start = time.time()
        while not os.path.exists(sync_lock):
            if time.time() - wait_start > 300:  # 5-minute timeout
                print(f"[MOXING]   Timeout waiting for sync lock. Proceeding anyway.")
                break
            time.sleep(1)
        print(f"[MOXING] Device {get_device_id()}: Sync completed (waited {time.time()-wait_start:.1f}s)")

def _resolve_data_path():
    """
    Resolve data path with COCO-aware fallbacks.
    Returns local path where raw data should be stored.
    """
    # Priority order: data_path → coco_root → mindrecord_dir parent → /cache/dataset/raw
    paths = [
        getattr(config, 'data_path', None),
        getattr(config, 'coco_root', None),
        getattr(config, 'mindrecord_dir', None),
        '/cache/dataset/raw'
    ]
    
    for path in paths:
        if path:
            # For mindrecord_dir, go up one level to get dataset root
            if 'mindrecord' in str(path).lower():
                path = os.path.dirname(path)
            return path
    
    return '/cache/dataset/raw'  # Final fallback

def _resolve_output_path():
    """Resolve output path with fallbacks"""
    return getattr(config, 'output_path', '/cache/train_out')

def moxing_wrapper(pre_process=None, post_process=None):
    """
    Moxing decorator for automatic OBS ↔ local synchronization.
    Handles download before training and upload after training.
    """
    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            print("="*70)
            print("MODELARTS INTEGRATION: Starting Moxing synchronization")
            print("="*70)
            
            if getattr(config, 'enable_modelarts', False):
                # ===== STEP 1: DOWNLOAD RAW DATA =====
                data_url = getattr(config, 'data_url', None)
                if data_url and data_url.startswith(('s3://', 'obs://')):
                    local_data_path = _resolve_data_path()
                    print(f"[MOXING] Downloading dataset: {data_url} → {local_data_path}")
                    sync_data(data_url, local_data_path)
                    
                    # Auto-trigger COCO conversion if raw data detected (critical enhancement)
                    raw_input = os.path.join(local_data_path, 'input')
                    raw_anno = os.path.join(local_data_path, 'annotation')
                    coco_root = getattr(config, 'coco_root', '/cache/dataset/coco')
                    
                    if os.path.exists(raw_input) and not os.path.exists(os.path.join(coco_root, 'annotations')):
                        print("[MOXING]   Raw data detected but COCO structure missing.")
                        print("         Auto-triggering conversion pipeline...")
                        try:
                            from scripts.convert_to_coco import main as convert_main
                            convert_main()
                            print("[MOXING]  COCO conversion completed")
                        except Exception as e:
                            print(f"[MOXING]   COCO conversion failed (will proceed if mindrecord exists): {e}")
                else:
                    print("[MOXING] Skipping dataset download (no data_url configured)")
                
                # ===== STEP 2: DOWNLOAD PRE-TRAINED CHECKPOINT (optional) =====
                pre_trained = getattr(config, 'pre_trained', None)
                if pre_trained and pre_trained.startswith(('s3://', 'obs://')):
                    local_ckpt = '/cache/backbone/mobilenetv2.ckpt'
                    os.makedirs(os.path.dirname(local_ckpt), exist_ok=True)
                    print(f"[MOXING] Downloading pre-trained backbone: {pre_trained} → {local_ckpt}")
                    sync_data(pre_trained, local_ckpt)
                    config.pre_trained = local_ckpt  # Update config to local path
                
                # ===== STEP 3: SETUP OUTPUT DIRECTORY =====
                output_path = _resolve_output_path()
                os.makedirs(output_path, exist_ok=True)
                ms.set_context(save_graphs_path=os.path.join(output_path, str(get_rank_id())))
                
                # Update config with resolved paths (critical for train.py)
                config.device_num = get_device_num()
                config.device_id = get_device_id()
                config.output_path = output_path
                
                # Run user pre-process hook
                if pre_process:
                    print("[MOXING] Running pre_process hook...")
                    pre_process()
            
            # ===== STEP 4: EXECUTE TRAINING =====
            try:
                print("="*70)
                print("STARTING TRAINING FUNCTION")
                print("="*70)
                run_func(*args, **kwargs)
                print("="*70)
                print("TRAINING COMPLETED SUCCESSFULLY")
                print("="*70)
            except Exception as e:
                print("="*70)
                print(f"TRAINING FAILED: {type(e).__name__}: {e}")
                print("="*70)
                raise
            
            # ===== STEP 5: UPLOAD RESULTS =====
            if getattr(config, 'enable_modelarts', False):
                if post_process:
                    print("[MOXING] Running post_process hook...")
                    post_process()
                
                train_url = getattr(config, 'train_url', None)
                if train_url and train_url.startswith(('s3://', 'obs://')):
                    output_path = _resolve_output_path()
                    if os.path.exists(output_path) and os.listdir(output_path):
                        print(f"[MOXING] Uploading results: {output_path} → {train_url}")
                        sync_data(output_path, train_url, is_upload=True)
                        print(f"[MOXING]  Upload completed: {train_url}")
                    else:
                        print(f"[MOXING]   Skipping upload: output directory empty ({output_path})")
                else:
                    print("[MOXING] Skipping upload (no train_url configured)")
        
        return wrapped_func
    return wrapper
