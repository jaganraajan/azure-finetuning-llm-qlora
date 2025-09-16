"""
Evaluation metrics for healthcare QLoRA models.
"""


import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from datasets import Dataset

# Advanced metrics (rouge-score, evaluate, bleu) are not installed
# METRICS_AVAILABLE will always be False
METRICS_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)

class HealthcareEvaluator:
    """Evaluator for healthcare Q&A models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.logger = get_logger(__name__)
        
        # Initialize metrics if available
    # Advanced metrics are not available
    self.metrics_available = False
    logger.warning("Evaluation metrics not available. Install rouge-score and evaluate packages.")
    
    def evaluate_model(self, trainer, test_dataset: Dataset) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info("Running model evaluation...")
        
        predictions = []
        references = []
        
        # Generate predictions
        for i, example in enumerate(test_dataset):
            if i % 10 == 0:
                logger.info(f"Processing example {i}/{len(test_dataset)}")
            
            # Format input
            instruction = "You are a helpful healthcare assistant. Please provide accurate and helpful medical information."
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{example['input']}\n\n### Response:\n"
            
            # Generate prediction
            try:
                prediction = trainer.generate_response(prompt, max_new_tokens=128)
                predictions.append(prediction)
                references.append(example['output'])
            except Exception as e:
                logger.warning(f"Error generating prediction for example {i}: {e}")
                predictions.append("")
                references.append(example['output'])
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, references)
        
        # Calculate healthcare-specific metrics
        healthcare_metrics = self._calculate_healthcare_metrics(predictions, references)
        metrics.update(healthcare_metrics)
        
        # Create detailed results
        results = {
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
            "examples": [
                {
                    "input": example["input"],
                    "reference": ref,
                    "prediction": pred
                }
                for example, ref, pred in zip(test_dataset, references, predictions)
            ]
        }
        
        logger.info("Evaluation completed")
        return results
    
    def _calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate standard NLP metrics."""
        metrics = {}
        
        if not self.metrics_available:
            logger.warning("Metrics not available, calculating basic metrics only")
            metrics.update(self._calculate_basic_metrics(predictions, references))
            return metrics
        
        try:
            # ROUGE scores
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                for metric_name in rouge_scores:
                    rouge_scores[metric_name].append(scores[metric_name].fmeasure)
            
            for metric_name, scores in rouge_scores.items():
                metrics[metric_name] = np.mean(scores)
            
            # BLEU score
            try:
                bleu_result = self.bleu_metric.compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                metrics['bleu'] = bleu_result['bleu']
            except Exception as e:
                logger.warning(f"Could not calculate BLEU: {e}")
                metrics['bleu'] = 0.0
            
            # BERTScore
            try:
                bert_result = self.bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang="en"
                )
                metrics['bertscore_f1'] = np.mean(bert_result['f1'])
                metrics['bertscore_precision'] = np.mean(bert_result['precision'])
                metrics['bertscore_recall'] = np.mean(bert_result['recall'])
            except Exception as e:
                logger.warning(f"Could not calculate BERTScore: {e}")
                metrics['bertscore_f1'] = 0.0
                metrics['bertscore_precision'] = 0.0
                metrics['bertscore_recall'] = 0.0
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics.update(self._calculate_basic_metrics(predictions, references))
        
        return metrics
    
    def _calculate_basic_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate basic metrics when advanced metrics are not available."""
        metrics = {}
        
        # Average length metrics
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        metrics['avg_prediction_length'] = np.mean(pred_lengths)
        metrics['avg_reference_length'] = np.mean(ref_lengths)
        metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        
        # Simple overlap metrics
        word_overlaps = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(ref_words) > 0:
                overlap = len(pred_words & ref_words) / len(ref_words)
            else:
                overlap = 0
            
            word_overlaps.append(overlap)
        
        metrics['word_overlap'] = np.mean(word_overlaps)
        
        return metrics
    
    def _calculate_healthcare_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate healthcare-specific metrics."""
        metrics = {}
        
        # Healthcare keywords
        healthcare_keywords = [
            'symptoms', 'treatment', 'diagnosis', 'medication', 'condition',
            'disease', 'health', 'medical', 'doctor', 'patient', 'therapy',
            'prevention', 'risk', 'side effects', 'dosage', 'consultation'
        ]
        
        # Calculate keyword coverage
        pred_keyword_coverage = []
        ref_keyword_coverage = []
        
        for pred, ref in zip(predictions, references):
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            pred_keywords = sum(1 for kw in healthcare_keywords if kw in pred_lower)
            ref_keywords = sum(1 for kw in healthcare_keywords if kw in ref_lower)
            
            pred_keyword_coverage.append(pred_keywords)
            ref_keyword_coverage.append(ref_keywords)
        
        metrics['avg_healthcare_keywords_pred'] = np.mean(pred_keyword_coverage)
        metrics['avg_healthcare_keywords_ref'] = np.mean(ref_keyword_coverage)
        
        # Response appropriateness (basic heuristics)
        appropriate_responses = 0
        for pred in predictions:
            if (len(pred.strip()) > 10 and  # Not too short
                not any(inappropriate in pred.lower() for inappropriate in ['i don\'t know', 'error', 'undefined']) and  # Not error responses
                any(keyword in pred.lower() for keyword in healthcare_keywords)):  # Contains healthcare content
                appropriate_responses += 1
        
        metrics['response_appropriateness'] = appropriate_responses / len(predictions) if predictions else 0
        
        return metrics
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_path / "metrics.json", "w") as f:
            json.dump(results["metrics"], f, indent=2)
        
        # Save detailed results
        detailed_results = []
        for i, example in enumerate(results["examples"]):
            detailed_results.append({
                "id": i,
                "input": example["input"],
                "reference": example["reference"],
                "prediction": example["prediction"],
            })
        
        # Save as JSON
        with open(output_path / "detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(detailed_results)
        df.to_csv(output_path / "detailed_results.csv", index=False)
        
        # Save summary report
        self._generate_summary_report(results, output_path / "evaluation_report.md")
        
        logger.info(f"Results saved to {output_path}")
    
    def _generate_summary_report(self, results: Dict[str, Any], report_path: Path):
        """Generate a summary evaluation report."""
        metrics = results["metrics"]
        
        report = ["# Healthcare QLoRA Model Evaluation Report\n"]
        
        report.append("## Model Performance Metrics\n")
        
        # Standard NLP metrics
        if 'rouge1' in metrics:
            report.append("### Standard NLP Metrics")
            report.append(f"- ROUGE-1 F1: {metrics['rouge1']:.4f}")
            report.append(f"- ROUGE-2 F1: {metrics['rouge2']:.4f}")
            report.append(f"- ROUGE-L F1: {metrics['rougeL']:.4f}")
            report.append(f"- BLEU Score: {metrics['bleu']:.4f}")
            
            if 'bertscore_f1' in metrics:
                report.append(f"- BERTScore F1: {metrics['bertscore_f1']:.4f}")
                report.append(f"- BERTScore Precision: {metrics['bertscore_precision']:.4f}")
                report.append(f"- BERTScore Recall: {metrics['bertscore_recall']:.4f}")
            
            report.append("")
        
        # Basic metrics
        report.append("### Basic Metrics")
        if 'avg_prediction_length' in metrics:
            report.append(f"- Average Prediction Length: {metrics['avg_prediction_length']:.2f} words")
            report.append(f"- Average Reference Length: {metrics['avg_reference_length']:.2f} words")
            report.append(f"- Length Ratio: {metrics['length_ratio']:.2f}")
        
        if 'word_overlap' in metrics:
            report.append(f"- Word Overlap: {metrics['word_overlap']:.4f}")
        
        report.append("")
        
        # Healthcare-specific metrics
        report.append("### Healthcare-Specific Metrics")
        if 'avg_healthcare_keywords_pred' in metrics:
            report.append(f"- Avg Healthcare Keywords (Predictions): {metrics['avg_healthcare_keywords_pred']:.2f}")
            report.append(f"- Avg Healthcare Keywords (References): {metrics['avg_healthcare_keywords_ref']:.2f}")
        
        if 'response_appropriateness' in metrics:
            report.append(f"- Response Appropriateness: {metrics['response_appropriateness']:.2f}")
        
        report.append("")
        
        # Sample predictions
        report.append("## Sample Predictions\n")
        examples = results["examples"][:5]  # Show first 5 examples
        
        for i, example in enumerate(examples, 1):
            report.append(f"### Example {i}")
            report.append(f"**Input:** {example['input']}")
            report.append(f"**Reference:** {example['reference']}")
            report.append(f"**Prediction:** {example['prediction']}")
            report.append("")
        
        # Write report
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Evaluation report saved to {report_path}")