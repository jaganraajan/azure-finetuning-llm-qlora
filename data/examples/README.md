# Healthcare QLoRA Project - Example Dataset

This directory contains example healthcare Q&A datasets for testing and demonstration purposes.

## Files

- `sample_healthcare_qa.csv`: Sample healthcare questions and answers
- `sample_healthcare_qa.json`: Same data in JSON format

## Data Format

The dataset should have the following columns:
- `input`: The healthcare question
- `output`: The corresponding answer
- `source`: Source of the data (optional)

## Usage

Use this sample data to test the training pipeline:

```bash
python scripts/train.py --data-path ./data/examples/sample_healthcare_qa.csv
```

## Data Privacy

All sample data has been anonymized and does not contain any real patient information. In production, ensure proper PHI handling and HIPAA compliance.