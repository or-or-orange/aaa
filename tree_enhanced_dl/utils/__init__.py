"""
Configuration Parser and Validator
"""

import yaml
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    Parse and validate configuration files
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        
        # Validate config
        ConfigParser.validate_config(config)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]):
        """
        Validate configuration structure and values
        
        Args:
            config: Configuration dictionary
        """
        required_sections = ['data', 'tree', 'model', 'loss', 'training', 'system']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate data config
        data_config = config['data']
        required_data_keys = ['target_column']
        for key in required_data_keys:
            if key not in data_config:
                raise ValueError(f"Missing required data config key: {key}")
        
        # Validate tree config
        tree_config = config['tree']
        if tree_config['framework'] not in ['lightgbm', 'xgboost', 'catboost']:
            raise ValueError(f"Invalid tree framework: {tree_config['framework']}")
        
        # Validate model config
        model_config = config['model']
        if model_config['sequence_encoder']['type'] not in ['bilstm', 'transformer', 'gru']:
            raise ValueError(f"Invalid encoder type: {model_config['sequence_encoder']['type']}")
        
        if model_config['fusion']['type'] not in ['multi_head_attention', 'concat', 'gated']:
            raise ValueError(f"Invalid fusion type: {model_config['fusion']['type']}")
        
        # Validate training config
        training_config = config['training']
        if training_config['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
        
        if training_config['num_epochs'] <= 0:
            raise ValueError("Number of epochs must be positive")
        
        logger.info("Configuration validated successfully")
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            save_path: Path to save config
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge two configurations, with override taking precedence
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigParser.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged


if __name__ == "__main__":
    # Test config parser
    config = ConfigParser.load_config('../configs/default_config.yaml')
    print("Config loaded successfully")
    print(f"Tree framework: {config['tree']['framework']}")
    print(f"Encoder type: {config['model']['sequence_encoder']['type']}")
