"""
Logging utilities for Azure QLoRA Healthcare project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger

class Logger:
    """Custom logger class with enhanced functionality."""
    
    def __init__(self, name: str, level: str = "INFO", log_file: Optional[str] = None):
        """Initialize logger."""
        self.name = name
        self.level = level
        self.log_file = log_file
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger configuration."""
        # Remove default handler
        loguru_logger.remove()
        
        # Console handler
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level=self.level,
            colorize=True,
        )
        
        # File handler if specified
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            loguru_logger.add(
                self.log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                       "{name}:{function}:{line} - {message}",
                level=self.level,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
            )
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        loguru_logger.bind(name=self.name).info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        loguru_logger.bind(name=self.name).debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        loguru_logger.bind(name=self.name).warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        loguru_logger.bind(name=self.name).error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        loguru_logger.bind(name=self.name).critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        loguru_logger.bind(name=self.name).exception(message, **kwargs)

# Global logger instances
_loggers = {}

def get_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> Logger:
    """Get or create a logger instance."""
    global _loggers
    
    logger_key = f"{name}_{level}_{log_file or 'console'}"
    
    if logger_key not in _loggers:
        _loggers[logger_key] = Logger(name, level, log_file)
    
    return _loggers[logger_key]

def setup_azure_ml_logging():
    """Setup logging for Azure ML experiments."""
    try:
        from azureml.core import Run
        
        # Get current run context
        run = Run.get_context()
        
        if hasattr(run, 'log'):
            # Add Azure ML handler to loguru
            def azure_ml_sink(message):
                record = message.record
                run.log(f"log_{record['level'].name.lower()}", record['message'])
            
            loguru_logger.add(azure_ml_sink, level="INFO")
            
    except ImportError:
        # Azure ML not available
        pass
    except Exception as e:
        # Not in Azure ML context
        pass

def setup_wandb_logging(project_name: str = "azure-healthcare-qlora"):
    """Setup logging for Weights & Biases."""
    try:
        import wandb
        
        if not wandb.run:
            wandb.init(project=project_name)
        
        # Add wandb handler to loguru
        def wandb_sink(message):
            record = message.record
            wandb.log({
                f"log_{record['level'].name.lower()}": record['message'],
                "timestamp": record['time'].timestamp()
            })
        
        loguru_logger.add(wandb_sink, level="INFO")
        
    except ImportError:
        # wandb not available
        pass
    except Exception as e:
        # wandb not initialized properly
        pass