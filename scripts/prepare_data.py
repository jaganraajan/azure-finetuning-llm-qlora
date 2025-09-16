"""
Data preparation script for healthcare datasets.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from azure_qlora_healthcare.data.processor import HealthcareDataProcessor
from azure_qlora_healthcare.utils.logger import get_logger

def main():
    """Prepare healthcare dataset for training."""
    parser = argparse.ArgumentParser(description="Healthcare Data Preparation")
    parser.add_argument("--input-path", type=str, help="Path to input dataset")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--dataset-type", type=str, choices=["medqa", "custom"], default="medqa", help="Dataset type")
    parser.add_argument("--anonymize", action="store_true", help="Apply PHI anonymization")
    parser.add_argument("--validate", action="store_true", help="Validate dataset quality")
    
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    logger.info("Starting data preparation...")
    
    try:
        # Initialize data processor
        data_processor = HealthcareDataProcessor()
        
        # Load dataset
        if args.dataset_type == "medqa":
            logger.info("Loading MedQA dataset...")
            dataset = data_processor.load_medqa_dataset()
        elif args.input_path:
            logger.info(f"Loading custom dataset from {args.input_path}")
            dataset = data_processor.load_custom_dataset(args.input_path)
        else:
            logger.error("No input path specified for custom dataset")
            return
        
        # Validate dataset if requested
        if args.validate:
            logger.info("Validating dataset...")
            if not data_processor.validate_dataset(dataset):
                logger.error("Dataset validation failed")
                return
        
        # Format for training
        logger.info("Formatting dataset for training...")
        formatted_dataset = data_processor.format_for_training(dataset)
        
        # Save processed dataset
        logger.info(f"Saving processed dataset to {args.output_dir}")
        data_processor.save_dataset(formatted_dataset, args.output_dir)
        
        # Print summary
        total_examples = sum(len(split) for split in formatted_dataset.values())
        print(f"\n=== Data Preparation Summary ===")
        print(f"Total examples: {total_examples}")
        for split_name, split_data in formatted_dataset.items():
            print(f"{split_name}: {len(split_data)} examples")
        print(f"Output directory: {args.output_dir}")
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()