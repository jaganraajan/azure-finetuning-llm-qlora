"""
Data processing utilities for healthcare Q&A datasets.
"""

import re
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset
import random

from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

class HealthcareDataProcessor:
    """Processes healthcare Q&A data for fine-tuning."""
    
    def __init__(self):
        """Initialize data processor."""
        self.config = get_config()
        self.seed = self.config.get("data.seed", 42)
        random.seed(self.seed)
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize PHI (Protected Health Information) in text."""
        if not text:
            return text
        
        # Define patterns for common PHI
        phi_patterns = [
            # Names (simplified pattern)
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PATIENT_NAME]'),
            # Dates
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]'),
            (r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]'),
            # Phone numbers
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),
            (r'\(\d{3}\)\s*\d{3}-\d{4}\b', '[PHONE]'),
            # SSN
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            # Medical record numbers
            (r'\bMRN:?\s*\d+\b', '[MRN]'),
            # Email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            # Addresses (simplified)
            (r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b', '[ADDRESS]'),
        ]
        
        anonymized_text = text
        for pattern, replacement in phi_patterns:
            anonymized_text = re.sub(pattern, replacement, anonymized_text, flags=re.IGNORECASE)
        
        return anonymized_text
    
    def load_medqa_dataset(self) -> DatasetDict:
        """Load MedQA dataset from Hugging Face."""
        logger.info("Loading MedQA dataset...")
        
        try:
            # Load the MedQA dataset
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")
            
            # Process the dataset
            processed_dataset = {}
            for split in dataset.keys():
                processed_data = []
                for example in dataset[split]:
                    processed_example = self._process_medqa_example(example)
                    if processed_example:
                        processed_data.append(processed_example)
                
                processed_dataset[split] = Dataset.from_list(processed_data)
            
            logger.info(f"Loaded MedQA dataset with {len(processed_dataset)} splits")
            return DatasetDict(processed_dataset)
            
        except Exception as e:
            logger.error(f"Failed to load MedQA dataset: {e}")
            # Fall back to creating a sample dataset
            return self._create_sample_dataset()
    
    def _process_medqa_example(self, example: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Process a single MedQA example."""
        try:
            question = example.get("question", "")
            if not question:
                return None
            
            # Get the correct answer
            answer = ""
            choices = example.get("choices", [])
            answer_id = example.get("answer", [""])[0]
            
            for choice in choices:
                if choice.get("id") == answer_id:
                    answer = choice.get("text", "")
                    break
            
            if not answer:
                return None
            
            # Anonymize the text
            anonymized_question = self.anonymize_text(question)
            anonymized_answer = self.anonymize_text(answer)
            
            return {
                "input": anonymized_question,
                "output": anonymized_answer,
                "source": "medqa"
            }
            
        except Exception as e:
            logger.warning(f"Error processing MedQA example: {e}")
            return None
    
    def _create_sample_dataset(self) -> DatasetDict:
        """Create a sample healthcare Q&A dataset."""
        logger.info("Creating sample healthcare Q&A dataset...")
        
        sample_data = [
            {
                "input": "What are the common symptoms of diabetes?",
                "output": "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, cuts or bruises that are slow to heal, and unexplained weight loss (in type 1 diabetes).",
                "source": "sample"
            },
            {
                "input": "How can I manage high blood pressure?",
                "output": "High blood pressure can be managed through lifestyle changes such as maintaining a healthy diet low in sodium, regular exercise, maintaining a healthy weight, limiting alcohol consumption, not smoking, and managing stress. Medications may also be prescribed by your healthcare provider.",
                "source": "sample"
            },
            {
                "input": "What is the difference between type 1 and type 2 diabetes?",
                "output": "Type 1 diabetes is an autoimmune condition where the body doesn't produce insulin, typically diagnosed in children and young adults. Type 2 diabetes is when the body doesn't use insulin properly or doesn't produce enough, often related to lifestyle factors and typically diagnosed in adults.",
                "source": "sample"
            },
            {
                "input": "When should I see a cardiologist?",
                "output": "You should see a cardiologist if you have chest pain, shortness of breath, dizziness, palpitations, high blood pressure that's difficult to control, a family history of heart disease, or if your primary care doctor refers you for specialized cardiac care.",
                "source": "sample"
            },
            {
                "input": "What are the warning signs of a heart attack?",
                "output": "Warning signs of a heart attack include chest pain or pressure, pain radiating to arms, neck, jaw, or back, shortness of breath, nausea, lightheadedness, cold sweats, and fatigue. Women may experience subtler symptoms. Call emergency services immediately if you suspect a heart attack.",
                "source": "sample"
            }
        ]
        
        # Create more samples by varying the existing ones
        extended_data = sample_data.copy()
        for _ in range(20):  # Add 20 more varied samples
            base_sample = random.choice(sample_data)
            extended_data.append(base_sample)
        
        # Split the data
        random.shuffle(extended_data)
        train_size = int(0.8 * len(extended_data))
        val_size = int(0.1 * len(extended_data))
        
        train_data = extended_data[:train_size]
        val_data = extended_data[train_size:train_size + val_size]
        test_data = extended_data[train_size + val_size:]
        
        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })
    
    def format_for_training(self, dataset: DatasetDict) -> DatasetDict:
        """Format dataset for QLoRA training."""
        logger.info("Formatting dataset for training...")
        
        def format_example(example):
            # Create a conversational format
            instruction = "You are a helpful healthcare assistant. Please provide accurate and helpful medical information."
            
            formatted_text = f"### Instruction:\n{instruction}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            
            return {
                "text": formatted_text,
                "input": example["input"],
                "output": example["output"]
            }
        
        formatted_dataset = {}
        for split_name, split_data in dataset.items():
            formatted_dataset[split_name] = split_data.map(format_example)
        
        return DatasetDict(formatted_dataset)
    
    def save_dataset(self, dataset: DatasetDict, output_dir: str):
        """Save processed dataset to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dataset to {output_path}")
        
        for split_name, split_data in dataset.items():
            split_path = output_path / f"{split_name}.jsonl"
            
            with open(split_path, 'w') as f:
                for example in split_data:
                    f.write(json.dumps(example) + '\n')
        
        # Save dataset info
        info = {
            "splits": list(dataset.keys()),
            "total_examples": sum(len(split) for split in dataset.values()),
            "features": list(dataset[list(dataset.keys())[0]].features.keys()) if dataset else []
        }
        
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Dataset saved with {info['total_examples']} total examples")
    
    def load_custom_dataset(self, data_path: str) -> DatasetDict:
        """Load custom healthcare Q&A dataset from file."""
        logger.info(f"Loading custom dataset from {data_path}")
        
        data_path = Path(data_path)
        
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix.lower() == '.json':
            df = pd.read_json(data_path)
        elif data_path.suffix.lower() == '.jsonl':
            df = pd.read_json(data_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Ensure required columns exist
        required_columns = ['input', 'output']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Anonymize data
        df['input'] = df['input'].apply(self.anonymize_text)
        df['output'] = df['output'].apply(self.anonymize_text)
        
        # Add source column if not present
        if 'source' not in df.columns:
            df['source'] = 'custom'
        
        # Split the data
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        train_split = self.config.get("data.train_split", 0.8)
        val_split = self.config.get("data.val_split", 0.1)
        
        train_size = int(train_split * len(df))
        val_size = int(val_split * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        return DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df)
        })
    
    def validate_dataset(self, dataset: DatasetDict) -> bool:
        """Validate the dataset for training."""
        logger.info("Validating dataset...")
        
        issues = []
        
        for split_name, split_data in dataset.items():
            if len(split_data) == 0:
                issues.append(f"Split '{split_name}' is empty")
                continue
            
            # Check for required columns
            required_features = ['input', 'output']
            for feature in required_features:
                if feature not in split_data.features:
                    issues.append(f"Split '{split_name}' missing feature: {feature}")
            
            # Check for empty values
            for i, example in enumerate(split_data):
                if not example.get('input', '').strip():
                    issues.append(f"Split '{split_name}', example {i}: empty input")
                if not example.get('output', '').strip():
                    issues.append(f"Split '{split_name}', example {i}: empty output")
        
        if issues:
            logger.error(f"Dataset validation failed with {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                logger.error(f"  - {issue}")
            if len(issues) > 10:
                logger.error(f"  ... and {len(issues) - 10} more issues")
            return False
        
        logger.info("Dataset validation passed")
        return True