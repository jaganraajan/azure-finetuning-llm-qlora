"""
QLoRA training implementation for healthcare language model fine-tuning.
"""

import os
import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import DatasetDict
import wandb

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..data.processor import HealthcareDataProcessor

logger = get_logger(__name__)

@dataclass
class QLoRATrainingConfig:
    """Configuration for QLoRA training."""
    
    # Model configuration
    model_name: str = field(default="microsoft/DialoGPT-medium")
    max_length: int = field(default=512)
    
    # LoRA configuration
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training configuration
    batch_size: int = field(default=8)
    learning_rate: float = field(default=2e-4)
    num_epochs: int = field(default=3)
    warmup_ratio: float = field(default=0.1)
    weight_decay: float = field(default=0.01)
    
    # Quantization configuration
    load_in_4bit: bool = field(default=True)
    bnb_4bit_use_double_quant: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: torch.dtype = field(default=torch.bfloat16)

class QLoRATrainer:
    """QLoRA trainer for healthcare language models."""
    
    def __init__(self, config: Optional[QLoRATrainingConfig] = None):
        """Initialize QLoRA trainer."""
        self.config = config or self._load_config_from_settings()
        self.logger = get_logger(__name__)
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def _load_config_from_settings(self) -> QLoRATrainingConfig:
        """Load configuration from settings."""
        app_config = get_config()
        
        return QLoRATrainingConfig(
            model_name=app_config.get("model.base_model_name", "microsoft/DialoGPT-medium"),
            max_length=app_config.get("model.max_length", 512),
            lora_r=app_config.get("qlora.lora_r", 16),
            lora_alpha=app_config.get("qlora.lora_alpha", 32),
            lora_dropout=app_config.get("qlora.lora_dropout", 0.1),
            target_modules=app_config.get("qlora.target_modules", ["q_proj", "v_proj"]),
            batch_size=app_config.get("model.batch_size", 8),
            learning_rate=app_config.get("model.learning_rate", 2e-4),
            num_epochs=app_config.get("model.num_epochs", 3),
            warmup_ratio=app_config.get("model.warmup_ratio", 0.1),
            weight_decay=app_config.get("model.weight_decay", 0.01),
        )
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with QLoRA configuration."""
        logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
    
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}"
        )
    
    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Preprocess dataset for training."""
        logger.info("Preprocessing dataset...")
        
        def tokenize_function(examples):
            # Tokenize the text
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None,
            )
            
            # For causal LM, labels are the same as input_ids
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
        )
        
        logger.info("Dataset preprocessing complete")
        return tokenized_dataset
    
    def setup_training_args(self, output_dir: str) -> TrainingArguments:
        """Setup training arguments."""
        app_config = get_config()
        
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=app_config.get("model.gradient_accumulation_steps", 1),
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=app_config.get("training.logging_steps", 100),
            save_strategy=app_config.get("training.save_strategy", "epoch"),
            evaluation_strategy=app_config.get("training.evaluation_strategy", "epoch"),
            save_total_limit=app_config.get("training.save_total_limit", 3),
            load_best_model_at_end=app_config.get("training.load_best_model_at_end", True),
            metric_for_best_model=app_config.get("training.metric_for_best_model", "eval_loss"),
            greater_is_better=app_config.get("training.greater_is_better", False),
            dataloader_num_workers=app_config.get("training.dataloader_num_workers", 4),
            remove_unused_columns=app_config.get("training.remove_unused_columns", False),
            fp16=torch.cuda.is_available(),
            report_to="wandb" if wandb.run else "none",
            run_name=f"healthcare-qlora-{app_config.get('model.base_model_name', 'model').split('/')[-1]}",
        )
    
    def train(
        self, 
        dataset: DatasetDict, 
        output_dir: str = "./outputs",
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the model using QLoRA."""
        logger.info("Starting QLoRA training...")
        
        # Setup model and tokenizer if not already done
        if self.model is None:
            self.setup_model_and_tokenizer()
        
        # Preprocess dataset
        tokenized_dataset = self.preprocess_dataset(dataset)
        
        # Setup training arguments
        training_args = self.setup_training_args(output_dir)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.trainer.save_state()
        
        logger.info("Training completed successfully")
        
        return {
            "training_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
        }
    
    def evaluate(self, dataset: Optional[DatasetDict] = None) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info("Evaluating model...")
        
        eval_results = self.trainer.evaluate()
        
        logger.info("Evaluation completed")
        return eval_results
    
    def save_model(self, output_dir: str):
        """Save the trained model and tokenizer."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info(f"Saving model to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        import json
        config_dict = {
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "max_length": self.config.max_length,
        }
        
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved successfully to {output_dir}")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        from peft import PeftModel
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        logger.info("Model loaded successfully")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate response for a given prompt."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response