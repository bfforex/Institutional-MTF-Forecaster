"""
Configuration Loading Utilities
"""

import yaml
from typing import Dict, Any
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to config YAML file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs(config_dir: str = 'configs') -> Dict[str, Dict]:
    """
    Load all configuration files from directory.
    
    Parameters
    ----------
    config_dir : str
        Directory containing config files
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping config name to config dict
    """
    configs = {}
    
    config_files = {
        'features_mtf': 'features_mtf.yaml',
        'model': 'model_base.yaml',
        'training': 'training.yaml',
        'backtest': 'backtest.yaml'
    }
    
    for name, filename in config_files.items():
        filepath = os.path.join(config_dir, filename)
        if os.path.exists(filepath):
            configs[name] = load_config(filepath)
        else:
            print(f"Warning: Config file not found: {filepath}")
    
    return configs


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    config_path : str
        Path to save config file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved to {config_path}")


def merge_configs(*configs: Dict) -> Dict:
    """
    Merge multiple configuration dictionaries.
    
    Later configs override earlier ones.
    
    Parameters
    ----------
    *configs : Dict
        Variable number of config dictionaries
        
    Returns
    -------
    Dict
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        _deep_merge(merged, config)
    
    return merged


def _deep_merge(base: Dict, update: Dict):
    """Recursively merge update into base."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
