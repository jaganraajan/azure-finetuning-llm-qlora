"""
Main training script for healthcare QLoRA fine-tuning.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from azure_qlora_healthcare.data.processor import HealthcareDataProcessor
from azure_qlora_healthcare.training.qlora_trainer import QLoRATrainer
from azure_qlora_healthcare.utils.azure_ml import AzureMLManager
from azure_qlora_healthcare.utils.logger import get_logger, setup_azure_ml_logging, setup_wandb_logging
from azure_qlora_healthcare.utils.config import get_config

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Healthcare QLoRA Fine-tuning")
    parser.add_argument("--data-path", type=str, help="Path to custom dataset")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model-name", type=str, help="Base model name")
    parser.add_argument("--use-azure-ml", action="store_true", help="Use Azure ML for training")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(__name__)
    logger.info("Starting healthcare QLoRA fine-tuning")
    
    # Setup monitoring
    if args.wandb_project:
        setup_wandb_logging(args.wandb_project)
    
    if args.use_azure_ml:
        setup_azure_ml_logging()
    
    try:
        # Load configuration
        config = get_config()
        
        # Validate configuration
        if args.use_azure_ml:
            config.validate_required_configs()
        
        # Initialize data processor
        logger.info("Initializing data processor...")
        data_processor = HealthcareDataProcessor()
        
        # Load and process dataset
        if args.data_path:
            logger.info(f"Loading custom dataset from {args.data_path}")
            dataset = data_processor.load_custom_dataset(args.data_path)
        else:
            logger.info("Loading MedQA dataset...")
            dataset = data_processor.load_medqa_dataset()
        
        # Validate dataset
        if not data_processor.validate_dataset(dataset):
            logger.error("Dataset validation failed")
            return
        
        # Format dataset for training
        dataset = data_processor.format_for_training(dataset)
        
        # Save processed dataset
        data_output_dir = Path(args.output_dir) / "data"
        data_processor.save_dataset(dataset, str(data_output_dir))
        
        # Initialize trainer
        logger.info("Initializing QLoRA trainer...")
        trainer = QLoRATrainer()
        
        if args.model_name:
            trainer.config.model_name = args.model_name
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        if args.use_azure_ml:
            # Use Azure ML for training
            logger.info("Starting Azure ML training...")
            azure_ml = AzureMLManager()
            
            # Submit training job
            job = azure_ml.submit_training_job(
                training_script="scripts/train.py",
                experiment_name="healthcare-qlora-training"
            )
            
            logger.info(f"Azure ML job submitted: {job.name}")
            logger.info(f"Job status: {job.status}")
            
        else:
            # Local training
            logger.info("Starting local training...")
            
            # Train model
            training_results = trainer.train(
                dataset=dataset,
                output_dir=args.output_dir,
                resume_from_checkpoint=args.resume_from_checkpoint
            )
            
            logger.info("Training completed!")
            logger.info(f"Training results: {training_results}")
            
            # Evaluate model
            if "validation" in dataset:
                eval_results = trainer.evaluate()
                logger.info(f"Evaluation results: {eval_results}")
            
            # Save model
            model_output_dir = Path(args.output_dir) / "model"
            trainer.save_model(str(model_output_dir))
            
            logger.info(f"Model saved to {model_output_dir}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()