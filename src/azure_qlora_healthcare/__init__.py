"""
Azure QLoRA Healthcare Fine-tuning Package

A comprehensive package for fine-tuning large language models using QLoRA 
technique for healthcare Q&A applications, integrated with Azure services.
"""

__version__ = "0.1.0"
__author__ = "Healthcare AI Team"
__email__ = "team@healthcareai.com"

from .utils.logger import get_logger
from .utils.config import load_config

__all__ = ["get_logger", "load_config"]