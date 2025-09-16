"""
Evaluation script for healthcare QLoRA model.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from azure_qlora_healthcare.training.qlora_trainer import QLoRATrainer
from azure_qlora_healthcare.data.processor import HealthcareDataProcessor
from azure_qlora_healthcare.evaluation.metrics import HealthcareEvaluator
from azure_qlora_healthcare.utils.logger import get_logger

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Healthcare QLoRA Model Evaluation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-path", type=str, help="Path to test dataset")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    logger.info("Starting model evaluation")
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        trainer = QLoRATrainer()
        trainer.load_model(args.model_path)
        
        # Load test data
        if args.data_path:
            logger.info(f"Loading test data from {args.data_path}")
            data_processor = HealthcareDataProcessor()
            dataset = data_processor.load_custom_dataset(args.data_path)
            test_data = dataset["test"]
        else:
            logger.info("Loading sample test data")
            data_processor = HealthcareDataProcessor()
            dataset = data_processor.load_medqa_dataset()
            test_data = dataset.get("test", dataset.get("validation"))
        
        if test_data is None:
            logger.error("No test data available")
            return
        
        # Limit samples if specified
        if args.max_samples:
            test_data = test_data.select(range(min(args.max_samples, len(test_data))))
        
        logger.info(f"Evaluating on {len(test_data)} samples")
        
        # Initialize evaluator
        evaluator = HealthcareEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_model(trainer, test_data)
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        evaluator.save_results(results, str(output_path))
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print("\n=== Evaluation Results ===")
        for metric, value in results["metrics"].items():
            print(f"{metric}: {value:.4f}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()