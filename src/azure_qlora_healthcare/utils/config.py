"""
Configuration management for Azure QLoRA Healthcare project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
        self._load_env_variables()
        self._load_config_file()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        project_root = Path(__file__).parent.parent.parent.parent
        return str(project_root / "config" / "config.yaml")
    
    def _load_env_variables(self):
        """Load environment variables."""
        # Load from .env file if it exists
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Azure configuration
        self.config.update({
            "azure": {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                "resource_group": os.getenv("AZURE_RESOURCE_GROUP"),
                "ml_workspace": os.getenv("AZURE_ML_WORKSPACE"),
                "location": os.getenv("AZURE_LOCATION", "eastus2"),
            },
            "compute": {
                "cluster_name": os.getenv("COMPUTE_CLUSTER_NAME", "gpu-cluster"),
                "vm_size": os.getenv("COMPUTE_VM_SIZE", "Standard_NC6s_v3"),
                "min_nodes": int(os.getenv("COMPUTE_MIN_NODES", "0")),
                "max_nodes": int(os.getenv("COMPUTE_MAX_NODES", "4")),
            },
            "model": {
                "base_model_name": os.getenv("BASE_MODEL_NAME", "microsoft/DialoGPT-medium"),
                "max_length": int(os.getenv("MAX_LENGTH", "512")),
                "batch_size": int(os.getenv("BATCH_SIZE", "8")),
                "learning_rate": float(os.getenv("LEARNING_RATE", "2e-4")),
                "num_epochs": int(os.getenv("NUM_EPOCHS", "3")),
            },
            "qlora": {
                "lora_r": int(os.getenv("LORA_R", "16")),
                "lora_alpha": int(os.getenv("LORA_ALPHA", "32")),
                "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.1")),
                "target_modules": os.getenv("TARGET_MODULES", "q_proj,v_proj").split(","),
            },
            "data": {
                "dataset_name": os.getenv("DATASET_NAME", "healthcare_qa"),
                "train_split": float(os.getenv("TRAIN_SPLIT", "0.8")),
                "val_split": float(os.getenv("VAL_SPLIT", "0.1")),
                "test_split": float(os.getenv("TEST_SPLIT", "0.1")),
            },
            "bot": {
                "app_id": os.getenv("BOT_APP_ID"),
                "app_password": os.getenv("BOT_APP_PASSWORD"),
                "endpoint": os.getenv("BOT_ENDPOINT"),
            },
            "monitoring": {
                "wandb_project": os.getenv("WANDB_PROJECT", "azure-healthcare-qlora"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
            },
            "health_data": {
                "fhir_service_url": os.getenv("FHIR_SERVICE_URL"),
                "workspace": os.getenv("HEALTH_DATA_WORKSPACE"),
            }
        })
    
    def _load_config_file(self):
        """Load configuration from YAML file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_configs(self.config, file_config)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def validate_required_configs(self) -> bool:
        """Validate that required configurations are present."""
        required_configs = [
            "azure.subscription_id",
            "azure.resource_group",
            "azure.ml_workspace",
        ]
        
        missing = []
        for config_key in required_configs:
            if not self.get(config_key):
                missing.append(config_key)
        
        if missing:
            raise ValueError(f"Missing required configurations: {missing}")
        
        return True

# Global configuration instance
_config = None

def load_config(config_path: Optional[str] = None) -> Config:
    """Load and return global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

def get_config() -> Config:
    """Get the global configuration instance."""
    if _config is None:
        return load_config()
    return _config