"""
Configuration module - Handles loading and validating configuration files
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """
    Centralized configuration manager for the RAG system.
    Loads settings from config.yaml and environment variables.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_env()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_env(self):
        """Load environment variables from .env file"""
        load_dotenv()
        
        # Store commonly used env vars
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.wandb_api_key = os.getenv("WANDB_API_KEY")
    
    def _validate_config(self):
        """Validate required configuration fields"""
        required_sections = ['paths', 'indexing', 'retrieval', 'generation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'indexing.embedding.model_name')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'retrieval.top_k')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str = None):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration (default: original config_path)
        """
        save_path = output_path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)


# Global config instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get or create global configuration instance (Singleton pattern).
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
