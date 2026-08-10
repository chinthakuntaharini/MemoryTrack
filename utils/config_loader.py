"""
Configuration loader for MemoryTrack system.
Loads and validates YAML configuration files with environment variable support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default to config/settings.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Replace environment variables
        self.config = self._replace_env_vars(self.config)
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def _replace_env_vars(self, config: Any) -> Any:
        """
        Recursively replace environment variables in configuration.
        
        Args:
            config: Configuration value (dict, list, or primitive)
            
        Returns:
            Configuration with environment variables replaced
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            default_value = None
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)
            return os.environ.get(env_var, default_value)
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).
        
        Args:
            key: Configuration key (e.g., "detection.confidence_threshold")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dots).
        
        Args:
            key: Configuration key (e.g., "detection.confidence_threshold")
            value: Value to set
        """
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def validate(self) -> bool:
        """
        Validate configuration structure and values.
        
        Returns:
            True if configuration is valid
        """
        required_sections = [
            "detection",
            "tracking",
            "feature_extraction",
            "memory_bank",
            "fusion_weights"
        ]
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate detection settings
        if "device" not in self.config["detection"]:
            logger.warning("Detection device not specified, defaulting to 'cpu'")
            self.config["detection"]["device"] = "cpu"
        
        # Validate memory bank settings
        if "embedding_dim" not in self.config["memory_bank"]:
            logger.error("Memory bank embedding_dim not specified")
            return False
        
        # Validate fusion weights sum to approximately 1.0
        weights = self.config.get("fusion_weights", {})
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.1:
            logger.warning(f"Fusion weights sum to {total_weight}, normalizing")
            for key in weights:
                weights[key] = weights[key] / total_weight
        
        logger.info("Configuration validation passed")
        return True
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.config.get("detection", {})
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration."""
        return self.config.get("tracking", {})
    
    def get_feature_extraction_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return self.config.get("feature_extraction", {})
    
    def get_memory_bank_config(self) -> Dict[str, Any]:
        """Get memory bank configuration."""
        return self.config.get("memory_bank", {})
    
    def get_fusion_weights(self) -> Dict[str, float]:
        """Get feature fusion weights."""
        return self.config.get("fusion_weights", {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get("database", {})
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard configuration."""
        return self.config.get("dashboard", {})
    
    def __repr__(self) -> str:
        return f"ConfigLoader(config_path={self.config_path})"
