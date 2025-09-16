"""
Azure setup script for initializing Azure ML workspace and resources.
"""

import sys
from pathlib import Path
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.azure_qlora_healthcare.utils.azure_ml import AzureMLManager
from src.azure_qlora_healthcare.utils.logger import get_logger
from src.azure_qlora_healthcare.utils.config import get_config

def main():
    """Setup Azure ML workspace and resources."""
    logger = get_logger(__name__)
    logger.info("Setting up Azure ML workspace...")
    
    try:
        # Load configuration
        config = get_config()
        
        # Validate required configurations
        config.validate_required_configs()
        
        # Initialize Azure ML manager
        azure_ml = AzureMLManager()
        
        # Create compute cluster
        logger.info("Creating compute cluster...")
        compute = azure_ml.create_compute_cluster()
        logger.info(f"Compute cluster ready: {compute.name}")
        
        # Create environment
        logger.info("Creating training environment...")
        environment = azure_ml.create_environment()
        logger.info(f"Environment ready: {environment.name}")
        
        logger.info("Azure ML setup completed successfully!")
        
        print("\n=== Azure ML Setup Summary ===")
        print(f"Workspace: {config.get('azure.ml_workspace')}")
        print(f"Resource Group: {config.get('azure.resource_group')}")
        print(f"Compute Cluster: {compute.name}")
        print(f"Environment: {environment.name}")
        print("\nYou can now run training with --use-azure-ml flag")
        
    except Exception as e:
        logger.error(f"Azure ML setup failed: {e}")
        print("\nSetup failed. Please check your Azure credentials and configuration.")
        raise

if __name__ == "__main__":
    main()