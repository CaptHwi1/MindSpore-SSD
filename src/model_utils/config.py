#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MindSpore SSD Configuration Loader
Enhanced for ModelArts GPU training + Local debugging with path resolution & safety checks
"""
import os
import sys
import ast
import argparse
from pathlib import Path
from pprint import pformat, pprint
import yaml

class Config:
    """Recursive configuration object with attribute access"""
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)
    
    def __str__(self):
        return pformat(self.__dict__)
    
    def __repr__(self):
        return self.__str__()
    
    def get(self, key, default=None):
        """Safe attribute access with default fallback"""
        return getattr(self, key, default)

def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="ssd300_config_gpu.yaml"):
    """
    Parse CLI arguments with YAML defaults.
    CRITICAL FIX: Accepts unknown args (ModelArts injects --data_url/--train_url)
    """
    parser = argparse.ArgumentParser(
        description="MindSpore SSD Configuration", 
        parents=[parser],
        allow_abbrev=False  # Prevent ambiguous arg matching
    )
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    
    # Register all YAML parameters as CLI args
    for item in cfg:
        if not isinstance(cfg[item], (list, dict)):
            help_description = helper.get(item, f"From {cfg_path}")
            choice = choices.get(item)
            
            if isinstance(cfg[item], bool):
                # Handle bool args properly (--flag=True/False)
                parser.add_argument(
                    f"--{item}", 
                    type=lambda x: (str(x).lower() == 'true'),
                    default=cfg[item],
                    choices=choice,
                    help=help_description
                )
            else:
                parser.add_argument(
                    f"--{item}", 
                    type=type(cfg[item]) if cfg[item] is not None else str,
                    default=cfg[item],
                    choices=choice,
                    help=help_description
                )
    
    # CRITICAL FIX: Accept unknown args (ModelArts injects extra params)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[CONFIG] Ignored unknown CLI arguments: {unknown}", file=sys.stderr)
    
    return args

def parse_yaml(yaml_path):
    """Safely parse YAML configuration file"""
    yaml_path = Path(yaml_path).expanduser().resolve()
    if not yaml_path.exists():
        # Fallback to common locations
        fallback_paths = [
            Path("./config/ssd300_config_gpu.yaml"),
            Path("../config/ssd300_config_gpu.yaml"),
            Path("../../config/ssd300_config_gpu.yaml"),
            Path("./ssd300_config_gpu.yaml")
        ]
        for fp in fallback_paths:
            if fp.exists():
                print(f"[CONFIG] Primary config not found, using fallback: {fp}", file=sys.stderr)
                yaml_path = fp.resolve()
                break
        else:
            raise FileNotFoundError(
                f"Configuration file not found at {yaml_path}\n"
                f"Tried fallback paths: {[str(p) for p in fallback_paths]}"
            )
    
    with open(yaml_path, 'r', encoding='utf-8') as fin:
        try:
            content = fin.read()
            cfgs = list(yaml.load_all(content, Loader=yaml.FullLoader))
            if len(cfgs) == 1:
                return cfgs[0], {}, {}
            elif len(cfgs) == 2:
                return cfgs[0], cfgs[1], {}
            elif len(cfgs) == 3:
                return cfgs[0], cfgs[1], cfgs[2]
            else:
                return cfgs[0], {}, {}
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing failed in {yaml_path}:\n{e}")
        except Exception as e:
            raise ValueError(f"Failed to parse config file {yaml_path}:\n{type(e).__name__}: {e}")

def resolve_paths(cfg_dict):
    """
    Auto-resolve path attributes for ModelArts vs Local environments.
    Handles: ~ expansion, /cache fallback, and directory creation.
    """
    # Path-like attributes to resolve
    path_attrs = [
        'coco_root', 'mindrecord_dir', 'output_path', 'checkpoint_path',
        'data_path', 'voc_root', 'pre_trained', 'feature_extractor_base_param'
    ]
    
    # Detect environment
    is_modelarts = cfg_dict.get('enable_modelarts', False)
    if not is_modelarts and 'MA_JOB_DIR' in os.environ:
        is_modelarts = True
    
    for attr in path_attrs:
        if attr in cfg_dict and cfg_dict[attr]:
            raw_path = cfg_dict[attr]
            
            # Skip OBS/s3 URLs
            if str(raw_path).startswith(('s3://', 'obs://')):
                continue
            
            # Expand user (~) for local paths
            if not is_modelarts:
                resolved = Path(str(raw_path)).expanduser().resolve()
            else:
                # ModelArts: Force /cache for performance-critical paths
                if attr in ['coco_root', 'mindrecord_dir', 'output_path', 'checkpoint_path']:
                    resolved = Path(f"/cache/dataset/{attr.replace('_root', '').replace('_dir', '').replace('_path', '')}")
                else:
                    resolved = Path(str(raw_path)).resolve()
            
            # Create parent directories
            try:
                if attr.endswith(('dir', 'path', 'root')):
                    resolved.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[CONFIG] Warning: Could not create {attr} directory {resolved}: {e}", file=sys.stderr)
            
            cfg_dict[attr] = str(resolved)
            print(f"[CONFIG] Resolved {attr:20s} → {resolved}")
    
    return cfg_dict

def get_config():
    """Main configuration loader with environment-aware resolution"""
    parser = argparse.ArgumentParser(description="MindSpore SSD Config Loader", add_help=False)
    
    # Find config file location
    current_dir = Path(__file__).parent.resolve()
    root_dir = current_dir.parent.parent  # Assume src/model_utils/config.py structure
    
    # Default config locations (in priority order)
    config_locations = [
        root_dir / "config" / "ssd300_config_gpu.yaml",
        root_dir / "ssd300_config_gpu.yaml",
        Path("./config/ssd300_config_gpu.yaml").resolve(),
        Path("./ssd300_config_gpu.yaml").resolve()
    ]
    
    default_yaml = next((p for p in config_locations if p.exists()), config_locations[0])
    
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=str(default_yaml),
        help="Path to YAML configuration file"
    )
    
    # Parse config_path first (before full parsing)
    path_args, _ = parser.parse_known_args()
    config_path = Path(path_args.config_path).expanduser().resolve()
    
    print("="*70)
    print(f"LOADING CONFIGURATION: {config_path.name}")
    print(f"Location: {config_path}")
    print("="*70)
    
    # Load YAML content
    default, helper, choices = parse_yaml(config_path)
    
    # Resolve paths BEFORE CLI merging (critical for ModelArts)
    default = resolve_paths(default)
    
    # Parse CLI arguments with YAML defaults
    cli_args = parse_cli_to_yaml(
        parser=parser, 
        cfg=default, 
        helper=helper, 
        choices=choices, 
        cfg_path=str(config_path)
    )
    
    # Merge CLI args into config (CLI overrides YAML)
    for item in vars(cli_args):
        if item != 'config_path':  # Skip config_path itself
            default[item] = getattr(cli_args, item)
    
    # Final path resolution after CLI merge
    default = resolve_paths(default)
    
    # Safety checks for critical parameters
    required_params = ['model_name', 'dataset', 'num_classes', 'batch_size']
    missing = [p for p in required_params if p not in default]
    if missing:
        raise ValueError(f"Missing required config parameters: {missing}")
    
    # Auto-compute num_ssd_boxes if set to -1
    if default.get('num_ssd_boxes', 0) == -1:
        h, w = default.get('img_shape', [300, 300])
        steps = default.get('steps', [16, 32, 64, 100, 150, 300])
        num_default = default.get('num_default', [3, 6, 6, 6, 6, 6])
        num = sum((h // steps[i]) * (w // steps[i]) * num_default[i] for i in range(len(steps)))
        default['num_ssd_boxes'] = num
        print(f"[CONFIG] Auto-computed num_ssd_boxes = {num}")
    
    pprint(default)
    print("="*70)
    return Config(default)

# Global config instance
try:
    config = get_config()
except Exception as e:
    print(f"\n FATAL CONFIGURATION ERROR:\n{e}\n", file=sys.stderr)
    print(" TROUBLESHOOTING TIPS:", file=sys.stderr)
    print("   1. Verify ssd300_config_gpu.yaml exists in config/ directory", file=sys.stderr)
    print("   2. Check YAML syntax at https://www.yamllint.com/", file=sys.stderr)
    print("   3. Ensure all required parameters are defined (model_name, dataset, etc.)", file=sys.stderr)
    sys.exit(1)

# Backward compatibility for older code expecting module-level attributes
for key, value in config.__dict__.items():
    globals()[key] = value
