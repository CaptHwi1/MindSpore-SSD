import os
import ast
import argparse
from pprint import pformat, pprint
import yaml

class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="ssd300_config_gpu.yaml"):
    parser = argparse.ArgumentParser(description="MindSpore SSD Configuration", parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], (list, dict)):
            help_description = helper[item] if item in helper else f"Reference to {cfg_path}"
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice, help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice, help=help_description)
    return parser.parse_args()

def parse_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found at {yaml_path}")
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = list(yaml.load_all(fin.read(), Loader=yaml.FullLoader))
            if len(cfgs) == 1: return cfgs[0], {}, {}
            elif len(cfgs) == 2: return cfgs[0], cfgs[1], {}
            elif len(cfgs) == 3: return cfgs[0], cfgs[1], cfgs[2]
        except Exception as e:
            raise ValueError(f"Failed to parse yaml: {e}")

def get_config():
    parser = argparse.ArgumentParser(description="Default Config", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Improvised: Look for the YAML in the root directory first
    root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    default_yaml = os.path.join(root_dir, "ssd300_config_gpu.yaml")
    
    parser.add_argument("--config_path", type=str, default=default_yaml, help="Config file path")
    path_args, _ = parser.parse_known_args()
    
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    
    # Merge CLI args into YAML defaults
    for item in vars(args):
        default[item] = getattr(args, item)
        
    pprint(default)
    return Config(default)

config = get_config()
