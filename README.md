# Azure Healthcare LLM Fine-tuning with QLoRA

This project demonstrates fine-tuning Large Language Models (LLMs) using QLoRA (Quantized Low-Rank Adaptation) for healthcare Q&A applications, integrated with Azure services including Azure ML, Azure Health Data Services, and Azure Bot Service.

## 🎯 Project Overview

This project showcases:
- **QLoRA Fine-tuning**: Efficient parameter-efficient fine-tuning of LLMs
- **Healthcare Domain**: Specialized for medical Q&A and healthcare information
- **Azure Integration**: Complete Azure ecosystem utilization
- **Production Ready**: Bot deployment and health data compliance
- **Data Privacy**: PHI anonymization and HIPAA compliance considerations

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Azure ML       │    │   Deployment    │
│                 │    │                  │    │                 │
│ • Medical Q&A   │───▶│ • QLoRA Training │───▶│ • Azure Bot     │
│ • Synthetic EMR │    │ • Model Registry │    │ • Health APIs   │
│ • FHIR Data     │    │ • Compute Cluster│    │ • Web Interface │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Azure subscription with appropriate permissions
- CUDA-capable GPU (for local training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jaganraajan/azure-finetuning-llm-qlora.git
cd azure-finetuning-llm-qlora
```

2. **Set up environment**
```bash
# Using conda (recommended)
conda env create -f config/conda_env.yaml
conda activate healthcare-qlora

# Or using pip
pip install -r requirements.txt
```

3. **Configure Azure credentials**
```bash
# Copy and edit environment file
cp .env.example .env

# Edit .env with your Azure credentials
# - AZURE_SUBSCRIPTION_ID
# - AZURE_RESOURCE_GROUP  
# - AZURE_ML_WORKSPACE
```

4. **Set up Azure ML workspace**
```bash
python scripts/setup_azure.py
```

### Basic Usage

#### 1. Training a Model

**Local Training:**
```bash
python scripts/train.py --output-dir ./outputs
```

**Azure ML Training:**
```bash
python scripts/train.py --use-azure-ml --output-dir ./outputs
```

**Custom Dataset:**
```bash
python scripts/train.py --data-path ./data/my_healthcare_qa.csv
```

#### 2. Evaluating a Model

```bash
python scripts/evaluate.py --model-path ./outputs/model --output-dir ./evaluation_results
```

#### 3. Running the Healthcare Bot

```bash
python scripts/run_bot.py --model-path ./outputs/model
```

## 📊 Features

### Data Processing
- **PHI Anonymization**: Automatic detection and anonymization of Protected Health Information
- **Multi-format Support**: CSV, JSON, JSONL data loading
- **Quality Validation**: Data validation and quality checks
- **Sample Datasets**: Built-in healthcare Q&A examples

### Model Training
- **QLoRA Implementation**: Memory-efficient fine-tuning with 4-bit quantization
- **Healthcare Optimization**: Domain-specific prompting and formatting
- **Azure ML Integration**: Scalable cloud training with GPU clusters
- **Monitoring**: Weights & Biases and Azure ML logging

### Evaluation
- **Comprehensive Metrics**: ROUGE, BLEU, BERTScore evaluation
- **Healthcare-Specific**: Domain relevance and appropriateness scoring
- **Detailed Reports**: Automated evaluation reports and analysis

### Deployment
- **Azure Bot Service**: Production-ready chat interface
- **Health Data Integration**: FHIR API compatibility
- **Safety Features**: Response filtering and medical disclaimers

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_ML_WORKSPACE=your-ml-workspace

# Model Configuration
BASE_MODEL_NAME=microsoft/DialoGPT-medium
MAX_LENGTH=512
BATCH_SIZE=8
LEARNING_RATE=2e-4

# QLoRA Configuration
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
```

### Advanced Configuration

Edit `config/config.yaml` for detailed settings:

```yaml
# Training configuration
training:
  save_strategy: "epoch"
  evaluation_strategy: "epoch"
  logging_steps: 100

# QLoRA specific settings
qlora:
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"
```

## 📚 Usage Examples

### Custom Training Script

```python
from azure_qlora_healthcare.training.qlora_trainer import QLoRATrainer
from azure_qlora_healthcare.data.processor import HealthcareDataProcessor

# Initialize components
data_processor = HealthcareDataProcessor()
trainer = QLoRATrainer()

# Load and process data
dataset = data_processor.load_medqa_dataset()
dataset = data_processor.format_for_training(dataset)

# Train model
trainer.setup_model_and_tokenizer()
results = trainer.train(dataset, output_dir="./my_model")
```

### Bot Integration

```python
from azure_qlora_healthcare.deployment.bot_service import HealthcareBotManager

# Initialize bot with trained model
bot_manager = HealthcareBotManager(model_path="./my_model")
bot = bot_manager.get_bot()

# Use in your application
# (see scripts/run_bot.py for complete example)
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/azure_qlora_healthcare --cov-report=html
```

## 📈 Monitoring and Logging

### Weights & Biases Integration

```bash
# Set up wandb
wandb login

# Train with logging
python scripts/train.py --wandb-project my-healthcare-project
```

### Azure ML Monitoring

Monitor training jobs in Azure ML Studio:
- Training metrics and loss curves
- Resource utilization
- Model artifacts and versions

## 🔒 Security and Compliance

### Data Privacy
- Automatic PHI detection and anonymization
- Configurable anonymization patterns
- HIPAA compliance considerations

### Access Control
- Azure AD integration
- Role-based access control
- Secure credential management

## 🗂️ Project Structure

```
azure-finetuning-llm-qlora/
├── src/azure_qlora_healthcare/     # Main package
│   ├── data/                       # Data processing
│   ├── training/                   # QLoRA training
│   ├── evaluation/                 # Model evaluation
│   ├── deployment/                 # Bot deployment
│   └── utils/                      # Utilities
├── scripts/                        # Training and utility scripts
├── config/                         # Configuration files
├── notebooks/                      # Jupyter notebooks
├── tests/                          # Test suite
├── docs/                          # Documentation
└── data/                          # Data directory
```

## 📖 Documentation

- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Azure Setup](docs/azure_setup.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for educational and demonstration purposes. The AI model should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

## 🔗 Related Resources

- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure Bot Service Documentation](https://docs.microsoft.com/en-us/azure/bot-service/)
- [Azure Health Data Services](https://docs.microsoft.com/en-us/azure/healthcare-apis/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)