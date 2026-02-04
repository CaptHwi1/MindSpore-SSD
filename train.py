#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSD300 + MobileNetV2 Training Script
Optimized for ModelArts GPU + Local Debugging with Frugal Factory Dataset
"""
import os
import sys
import shutil
from pathlib import Path
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed, dtype
from src.ssd import SSD300, SsdInferWithDecoder, SSDWithLossCell, TrainingWrapper, ssd_mobilenet_v2
from src.dataset import create_ssd_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter_by_list
from src.eval_callback import EvalCallBack
from src.eval_utils import apply_eval
from src.box_utils import default_boxes
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

set_seed(1234)  # Reproducible seed

class OBSUploadCallback(Callback):
    """Upload checkpoints to OBS after each save (ModelArts only)"""
    def __init__(self, local_ckpt_dir, obs_train_url):
        super().__init__()
        self.local_ckpt_dir = Path(local_ckpt_dir)
        self.obs_train_url = obs_train_url.rstrip('/') + '/'
        self.uploaded_epochs = set()
    
    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        
        if epoch_num % config.save_checkpoint_epochs == 0 and epoch_num not in self.uploaded_epochs:
            if config.enable_modelarts and self.obs_train_url.startswith('s3://'):
                try:
                    import moxing as mox
                    ckpt_files = list(self.local_ckpt_dir.glob(f"ssd*.ckpt"))
                    if ckpt_files:
                        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
                        obs_path = f"{self.obs_train_url}checkpoint/ssd-{epoch_num:03d}.ckpt"
                        mox.file.copy(str(latest_ckpt), obs_path)
                        print(f"[ModelArts] Uploaded checkpoint to OBS: {obs_path}")
                        self.uploaded_epochs.add(epoch_num)
                except Exception as e:
                    print(f"[WARNING] Failed to upload checkpoint: {e}")

def resolve_data_paths():
    """Resolve dataset paths for ModelArts vs Local environments"""
    if config.enable_modelarts:
        # ModelArts: Use /cache for speed
        config.coco_root = "/cache/dataset/coco"
        config.mindrecord_dir = "/cache/dataset/mindrecord"
        config.output_path = "/cache/train_out"
        
        # Auto-download raw data from OBS if not in cache
        if not os.path.exists(config.coco_root) and hasattr(config, 'data_url'):
            try:
                import moxing as mox
                print(f"[ModelArts] Downloading dataset from OBS: {config.data_url} → /cache/dataset/raw")
                mox.file.copy_parallel(config.data_url, "/cache/dataset/raw")
                print("[ModelArts] Dataset download complete. Run conversion script separately to generate mindrecord.")
            except Exception as e:
                print(f"[WARNING] OBS download failed (will proceed if mindrecord exists): {e}")
    else:
        # Local: Expand user paths
        for attr in ['coco_root', 'mindrecord_dir', 'output_path']:
            if hasattr(config, attr):
                setattr(config, attr, str(Path(getattr(config, attr)).expanduser()))
    
    # Ensure critical directories exist
    for path in [config.mindrecord_dir, config.output_path, config.checkpoint_path]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Dataset root: {config.coco_root}")
    print(f"[INFO] MindRecord dir: {config.mindrecord_dir}")
    print(f"[INFO] Output path: {config.output_path}")

def validate_mindrecord_exists():
    """Check if mindrecord files exist before training"""
    mindrecord_pattern = os.path.join(config.mindrecord_dir, f"{config.mindrecord_file}*.mindrecord")
    import glob
    files = glob.glob(mindrecord_pattern)
    
    if not files:
        raise FileNotFoundError(
            f"No mindrecord files found at {mindrecord_pattern}\n"
            " SOLUTION: Run 'python create_mindrecord.py --config config/ssd300_config_gpu.yaml' first!"
        )
    
    print(f"[✓] Found {len(files)} mindrecord files: {files[:3]}")
    return files[0]  # Return first file path (dataset loader uses pattern internally)

def ssd_model_build():
    """Build SSD300 model with MobileNetV2 backbone"""
    if config.model_name == "ssd300":
        backbone = ssd_mobilenet_v2()
        ssd = SSD300(backbone=backbone, config=config)
        init_net_param(ssd)  # Initialize with Xavier/He
        
        # Optional: Freeze backbone layers
        if config.freeze_layer == "backbone":
            print("[INFO] Freezing MobileNetV2 backbone layers")
            for param in backbone.feature_1.trainable_params():
                param.requires_grad = False
            for param in backbone.feature_2.trainable_params():
                param.requires_grad = False
        
        # Load pre-trained MobileNetV2 backbone if specified
        if config.pre_trained and os.path.exists(config.pre_trained):
            print(f"[INFO] Loading pre-trained backbone from: {config.pre_trained}")
            param_dict = ms.load_checkpoint(config.pre_trained)
            
            # Filter out non-backbone parameters (SSD-specific layers)
            backbone_params = {k: v for k, v in param_dict.items() if 'backbone' in k or 'feature' in k}
            if backbone_params:
                ms.load_param_into_net(ssd.backbone, backbone_params, strict_load=False)
                print(f"[✓] Loaded {len(backbone_params)} backbone parameters")
            else:
                print("[WARNING] No backbone parameters found in checkpoint - initializing randomly")
    else:
        raise ValueError(f"Unsupported model_name: {config.model_name}. Only 'ssd300' supported for MobileNetV2.")
    
    return ssd

def set_graph_kernel_context(device_target, model):
    """Enable graph kernel optimizations for GPU"""
    if device_target == "GPU" and model == "ssd300":
        ms.set_context(enable_graph_kernel=True,
                       graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D,BatchNorm")

def set_ascend_pynative_mempool_block_size():
    """Ascend-specific memory optimization (not used for GPU)"""
    if ms.get_context("mode") == ms.PYNATIVE_MODE and config.device_target == "Ascend":
        ms.set_context(mempool_block_size="31GB")

@moxing_wrapper()
def train_net():
    # ============ PATH RESOLUTION ============
    resolve_data_paths()
    
    # ============ AUTO-COMPUTE num_ssd_boxes ============
    if hasattr(config, 'num_ssd_boxes') and config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
        config.num_ssd_boxes = num
        print(f"[INFO] Auto-computed num_ssd_boxes = {config.num_ssd_boxes}")

    # ============ DISTRIBUTED SETUP ============
    rank = 0
    device_num = 1
    if config.device_target == "CPU":
        ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
        loss_scale = 1.0
    else:
        ms.set_context(mode=ms.GRAPH_MODE, 
                      device_target=config.device_target, 
                      device_id=config.device_id,
                      save_graphs=False)  # Disable graph saving for speed
        
        set_graph_kernel_context(config.device_target, config.model_name)
        set_ascend_pynative_mempool_block_size()
        
        if config.run_distribute:
            device_num = config.device_num
            ms.reset_auto_parallel_context()
            ms.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, 
                                       gradients_mean=True,
                                       device_num=device_num)
            init()
            if config.all_reduce_fusion_config:
                ms.set_auto_parallel_context(all_reduce_fusion_config=config.all_reduce_fusion_config)
            rank = get_rank()

    # ============ MINDRECORD VALIDATION ============
    if config.only_create_dataset:
        print("[INFO] only_create_dataset=True - Skipping training, only validating mindrecord creation")
        mindrecord_file = create_mindrecord(config.dataset, 
                                           os.path.join(config.mindrecord_dir, config.mindrecord_file), 
                                           True)
        print(f" MindRecord creation complete: {mindrecord_file}")
        return
    
    # CRITICAL FIX: Validate mindrecord exists BEFORE dataset creation
    validate_mindrecord_exists()
    
    # ============ DATASET CREATION ============
    use_multiprocessing = (config.device_target != "CPU")
    dataset = create_ssd_dataset(
        os.path.join(config.mindrecord_dir, config.mindrecord_file),
        batch_size=config.batch_size,
        device_num=device_num,
        rank=rank,
        use_multiprocessing=use_multiprocessing,
        num_parallel_workers=config.num_parallel_workers
    )
    
    dataset_size = dataset.get_dataset_size()
    if dataset_size == 0:
        raise ValueError(
            "Dataset size is 0! Check:\n"
            "  1. MindRecord files exist in mindrecord_dir\n"
            "  2. Annotation JSON has valid image paths\n"
            "  3. Image files exist in coco_root/train2017/"
        )
    
    print(f" Dataset created successfully!")
    print(f"    - Samples: {dataset_size * config.batch_size}")
    print(f"    - Batches per epoch: {dataset_size}")
    print(f"    - Batch size: {config.batch_size}")
    print(f"    - Workers: {config.num_parallel_workers}")

    # ============ MODEL BUILD ============
    ssd = ssd_model_build()
    
    # FP16 conversion (critical for GPU memory)
    if getattr(config, 'use_float16', False):
        print("[INFO] Converting network to float16")
        ssd.to_float(dtype.float16)
    
    net = SSDWithLossCell(ssd, config)

    # ============ CHECKPOINT SETUP ============
    ckpt_save_dir = os.path.join(config.output_path, f'ckpt_rank_{rank}')
    Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
        keep_checkpoint_max=config.keep_checkpoint_max
    )
    ckpoint_cb = ModelCheckpoint(
        prefix=f"ssd_mobilenetv2_rank{rank}",
        directory=ckpt_save_dir,
        config=ckpt_config
    )
    
    # Load pre-trained weights if specified
    if config.pre_trained and os.path.exists(config.pre_trained):
        print(f"[INFO] Loading full model checkpoint: {config.pre_trained}")
        param_dict = ms.load_checkpoint(config.pre_trained)
        if config.filter_weight:
            filter_checkpoint_parameter_by_list(param_dict, config.checkpoint_filter_list)
        ms.load_param_into_net(net, param_dict, strict_load=False)
        print(" Checkpoint loaded successfully")

    # ============ LEARNING RATE SCHEDULE ============
    # CRITICAL FIX: Use lr_end instead of lr_end_rate * lr (lr is legacy VGG param)
    lr_end_value = getattr(config, 'lr_end', config.lr_init * 0.01)
    
    lr = Tensor(get_lr(
        global_step=config.pre_trained_epoch_size * dataset_size,
        lr_init=config.lr_init,
        lr_end=lr_end_value,
        lr_max=config.lr_init,  # For cosine decay, max = init
        warmup_epochs=config.lr_warmup_epochs,
        total_epochs=config.epoch_size,
        steps_per_epoch=dataset_size,
        lr_decay_mode=getattr(config, 'lr_decay_mode', 'cosine')
    ))
    
    print(f"[INFO] LR Schedule: {config.lr_decay_mode}")
    print(f"    - Init: {config.lr_init:.5f} | End: {lr_end_value:.6f} | Warmup: {config.lr_warmup_epochs} epochs")

    # ============ OPTIMIZER & WRAPPER ============
    opt = nn.Momentum(
        filter(lambda x: x.requires_grad, net.get_parameters()),
        learning_rate=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        loss_scale=config.loss_scale
    )
    
    net = TrainingWrapper(net, opt, config.loss_scale)

    # ============ CALLBACKS ============
    callbacks = [
        TimeMonitor(data_size=dataset_size),
        LossMonitor(per_print_times=config.loss_log_interval),
        ckpoint_cb
    ]
    
    # Add OBS upload callback for ModelArts
    if config.enable_modelarts and hasattr(config, 'train_url'):
        callbacks.append(OBSUploadCallback(ckpt_save_dir, config.train_url))
    
    # Add evaluation callback
    if config.run_eval:
        print("[INFO] Setting up evaluation callback")
        try:
            eval_net = SsdInferWithDecoder(ssd, Tensor(default_boxes), config)
            eval_net.set_train(False)
            
            # CRITICAL FIX: Proper COCO validation path construction
            if config.dataset == "coco":
                anno_json = os.path.join(
                    config.coco_root, 
                    "annotations", 
                    getattr(config, 'eval_annotation', 'instances_val2017.json')
                )
            else:
                raise ValueError(f"Eval only supported for COCO dataset (got {config.dataset})")
            
            # Create eval mindrecord if needed
            eval_mindrecord = os.path.join(
                config.mindrecord_dir, 
                getattr(config, 'eval_mindrecord_file', 'coco_val')
            )
            if not os.path.exists(eval_mindrecord + "0"):
                print("[WARNING] Eval mindrecord not found - creating now (slows first epoch)")
                create_mindrecord(config.dataset, eval_mindrecord, False)
            
            eval_dataset = create_ssd_dataset(
                eval_mindrecord,
                batch_size=config.batch_size,
                is_training=False,
                use_multiprocessing=False,
                num_parallel_workers=config.num_parallel_workers
            )
            
            eval_param_dict = {
                "net": eval_net,
                "dataset": eval_dataset,
                "anno_json": anno_json,
                "img_ids": list(range(eval_dataset.get_dataset_size() * config.batch_size))
            }
            
            eval_cb = EvalCallBack(
                apply_eval,
                eval_param_dict,
                interval=config.eval_interval,
                eval_start_epoch=getattr(config, 'eval_start_epoch', 1),
                save_best_ckpt=True,
                ckpt_directory=ckpt_save_dir,
                best_ckpt_name="best_map.ckpt",
                metrics_name="mAP"
            )
            callbacks.append(eval_cb)
            print(f"[✓] Evaluation configured: {anno_json}")
        except Exception as e:
            print(f"[WARNING] Failed to setup evaluation (training will continue): {e}")
            import traceback
            traceback.print_exc()

    # ============ TRAINING ============
    model = Model(net)
    dataset_sink_mode = (config.mode_sink == "sink" and config.device_target != "CPU")
    
    print("="*70)
    print(" STARTING TRAINING")
    print("="*70)
    print(f"Model: SSD300 + MobileNetV2")
    print(f"Dataset: {config.dataset} ({dataset_size} batches/epoch)")
    print(f"Epochs: {config.epoch_size} | Batch Size: {config.batch_size}")
    print(f"Device: {config.device_target} (Rank {rank}/{device_num})")
    print(f"Sink Mode: {'ENABLED' if dataset_sink_mode else 'DISABLED'}")
    print(f"Output Dir: {config.output_path}")
    print("="*70)
    
    try:
        model.train(
            config.epoch_size,
            dataset,
            callbacks=callbacks,
            dataset_sink_mode=dataset_sink_mode
        )
        print("\n TRAINING COMPLETED SUCCESSFULLY")
    except KeyboardInterrupt:
        print("\n TRAINING INTERRUPTED BY USER")
        raise
    except Exception as e:
        print(f"\n TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ============ MODELARTS CHECKPOINT SYNC ============
    if config.enable_modelarts and hasattr(config, 'train_url'):
        try:
            import moxing as mox
            print(f"\n[ModelArts] Syncing final checkpoints to OBS: {config.train_url}")
            mox.file.copy_parallel(config.output_path, config.train_url)
            print("[✓] Checkpoint sync complete")
        except Exception as e:
            print(f"[WARNING] Failed to sync checkpoints to OBS: {e}")

if __name__ == '__main__':
    # CRITICAL: Fix sys.path for local execution
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    try:
        train_net()
    except Exception as e:
        print(f"\n FATAL ERROR: {e}")
        sys.exit(1)
