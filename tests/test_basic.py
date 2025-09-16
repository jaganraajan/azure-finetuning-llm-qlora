"""
Basic tests for the healthcare QLoRA project.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that main modules can be imported."""
    try:
        from azure_qlora_healthcare.utils.config import Config
        from azure_qlora_healthcare.utils.logger import get_logger
        from azure_qlora_healthcare.data.processor import HealthcareDataProcessor
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_config_creation():
    """Test config creation."""
    from azure_qlora_healthcare.utils.config import Config
    
    config = Config()
    assert config is not None
    assert hasattr(config, 'config')
    assert isinstance(config.config, dict)

def test_logger_creation():
    """Test logger creation."""
    from azure_qlora_healthcare.utils.logger import get_logger
    
    logger = get_logger("test")
    assert logger is not None
    assert hasattr(logger, 'info')

def test_data_processor_creation():
    """Test data processor creation."""
    from azure_qlora_healthcare.data.processor import HealthcareDataProcessor
    
    processor = HealthcareDataProcessor()
    assert processor is not None
    assert hasattr(processor, 'anonymize_text')

def test_phi_anonymization():
    """Test PHI anonymization functionality."""
    from azure_qlora_healthcare.data.processor import HealthcareDataProcessor
    
    processor = HealthcareDataProcessor()
    
    # Test basic anonymization
    text_with_phi = "Patient John Doe, DOB 01/15/1980, called at 555-123-4567"
    anonymized = processor.anonymize_text(text_with_phi)
    
    # Should not contain the original name or phone
    assert "John Doe" not in anonymized
    assert "555-123-4567" not in anonymized
    assert "[PATIENT_NAME]" in anonymized or "[PHONE]" in anonymized

def test_sample_dataset_creation():
    """Test sample dataset creation."""
    from azure_qlora_healthcare.data.processor import HealthcareDataProcessor
    
    processor = HealthcareDataProcessor()
    dataset = processor._create_sample_dataset()
    
    assert dataset is not None
    assert "train" in dataset
    assert "validation" in dataset
    assert "test" in dataset
    assert len(dataset["train"]) > 0

if __name__ == "__main__":
    pytest.main([__file__])