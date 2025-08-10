# MLOps Pipeline for DistilBERT News Classification

A complete MLOps pipeline for fine-tuning DistilBERT on the AG News dataset with MLflow experiment tracking and AWS S3 storage. This system provides end-to-end automation for data preparation, model training, experiment tracking, and deployment.

## Overview

This pipeline implements a production-ready machine learning workflow that:

- Downloads and preprocesses the AG News dataset for 4-class text classification
- Fine-tunes DistilBERT model with configurable parameters
- Tracks experiments and metrics using MLflow
- Stores datasets and trained models in AWS S3
- Provides inference capabilities via AWS Lambda

The system is containerized using Docker and orchestrated with Docker Compose for easy deployment and scalability.

## Architecture

### Core Components

- **MLflow Service**: Containerized experiment tracking server with SQLite backend
- **Uploader Service**: Downloads AG News dataset and uploads to S3 storage
- **Trainer Service**: Fine-tunes DistilBERT model with MLflow tracking integration
- **Lambda Handler**: AWS Lambda function for model inference and predictions

### Data Flow

1. MLflow server starts and provides experiment tracking UI on port 5000
2. Uploader service downloads AG News dataset from Hugging Face and stores in S3 as JSONL files
3. Trainer service downloads dataset from S3, fine-tunes DistilBERT, logs to MLflow, and uploads model to S3
4. Lambda handler provides inference API by downloading models from S3 on demand

## Prerequisites

- Docker and Docker Compose installed
- AWS account with S3 access
- AWS credentials configured
- At least 4GB RAM for model training

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository and navigate to project directory
cd 8_mlops

# Copy environment template
cp .env.example .env

# Edit .env file with your AWS credentials
# Required variables:
# - AWS_ACCESS_KEY_ID=<your-access-key>
# - AWS_SECRET_ACCESS_KEY=<your-secret-key>
# - AWS_REGION=eu-central-1
# - S3_BUCKET=mlops-homework-<your-name>
```

### 2. Start MLflow Server

```bash
# Start MLflow tracking server
docker compose up mlflow -d

# Access MLflow UI at http://localhost:5000
```

### 3. Run the Pipeline

```bash
# Step 1: Upload dataset to S3 (REQUIRED - trainer depends on this)
docker compose --profile manual run --rm uploader

# Step 2: Train model with MLflow tracking (uses dataset from S3)
docker compose up trainer

# Monitor training progress in MLflow UI
```

## Detailed Usage

### MLflow Server Management

```bash
# Start MLflow server in background
docker compose up mlflow -d

# View MLflow logs
docker compose logs mlflow

# Stop MLflow server
docker compose down mlflow

# Access MLflow UI
open http://localhost:5000
```

### Dataset Upload

The uploader service downloads the AG News dataset from Hugging Face and uploads it to S3:

```bash
# Run uploader service (MUST run first)
docker compose --profile manual run --rm uploader

# Files uploaded to S3:
# s3://your-bucket/datasets/ag_news/train.jsonl
# s3://your-bucket/datasets/ag_news/test.jsonl  
# s3://your-bucket/datasets/ag_news/info.json
```

### Model Training

The trainer service downloads the dataset from S3 (uploaded by uploader), fine-tunes DistilBERT and tracks experiments:

```bash
# Run training with MLflow tracking (requires dataset in S3)
docker compose up trainer

# Alternative: Run as one-off container
docker compose run --rm trainer

# Training artifacts stored in:
# - MLflow: Experiments, metrics, parameters
# - S3: Final model files at s3://your-bucket/models/distilbert-agnews/
```

### Building Services Individually

```bash
# Build uploader service
docker build -t mlops-uploader ./uploader

# Build trainer service  
docker build -t mlops-trainer ./trainer

# Run built images
docker run --env-file .env mlops-uploader
docker run --env-file .env mlops-trainer
```

## Configuration

### Environment Variables

Required for all services:
- `AWS_ACCESS_KEY_ID`: AWS access key for S3 operations
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for S3 operations  
- `AWS_REGION`: AWS region (default: eu-central-1)
- `S3_BUCKET`: S3 bucket name for storing datasets and models

Training specific:
- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://localhost:5000)
- `SAMPLE_SIZE`: Number of training samples to use (default: 5000)

### MLflow Configuration

- **Backend Store**: SQLite database stored in `./mlflow/mlflow.db`
- **Artifact Store**: Local filesystem in `./mlflow/artifacts`
- **Server Port**: 5000 (accessible at http://localhost:5000)
- **Auto-logging**: Enabled for transformers library

## Project Structure

```
8_mlops/
├── README.md                   # Project documentation
├── CLAUDE.md                   # Claude Code guidance
├── docker-compose.yml          # Container orchestration
├── .env.example               # Environment template
├── .env                       # Local environment (create from template)
│
├── uploader/                  # Dataset upload service
│   ├── Dockerfile            # Container configuration
│   ├── requirements.txt      # Python dependencies
│   └── upload.py            # Dataset upload script
│
├── trainer/                   # Model training service
│   ├── Dockerfile            # Container configuration  
│   ├── requirements.txt      # Python dependencies
│   └── train.py             # Training script with MLflow
│
├── lambda/                    # Inference service
│   └── handler.py            # AWS Lambda function
│
└── mlflow/                    # MLflow data directory
    ├── mlflow.db             # Experiment database
    └── artifacts/            # Model artifacts
```

## AWS Lambda Deployment

### Lambda Function Setup

1. Package the lambda handler with dependencies
2. Create AWS Lambda function with Python 3.9 runtime
3. Configure environment variables (AWS_REGION, S3_BUCKET)
4. Set appropriate IAM permissions for S3 access
5. Configure API Gateway for HTTP access (optional)

### Inference API Usage

```python
# Lambda function expects JSON input:
{
    "text": "Apple reported strong quarterly earnings with iPhone sales up 15%"
}

# Returns prediction:
{
    "statusCode": 200,
    "body": {
        "predicted_class": "Business",
        "confidence": 0.95,
        "probabilities": {
            "World": 0.02,
            "Sports": 0.01, 
            "Business": 0.95,
            "Sci/Tech": 0.02
        }
    }
}
```

### Testing Lambda Locally

```python
# Test the handler locally
cd lambda
python handler.py

# Example output with test data
```

## Model Details

### DistilBERT Configuration
- **Base Model**: distilbert-base-uncased
- **Parameters**: ~66M parameters
- **Task**: Multi-class text classification
- **Classes**: 4 (World, Sports, Business, Sci/Tech)

### Training Parameters
- **Learning Rate**: 2e-5
- **Batch Size**: 8 (per device)
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Max Sequence Length**: 512 (with padding and truncation)

### Dataset Information
- **Source**: wangrongsheng/ag_news (downloaded by uploader from Hugging Face)
- **Training Samples**: 120K total (configurable subset via SAMPLE_SIZE)
- **Test Samples**: 7.6K total
- **Features**: Text content with corresponding labels
- **Format**: JSONL files stored in S3 and loaded by trainer service
- **Dependency**: Trainer service requires uploader to run first

## Troubleshooting

### Common Issues

**MLflow Connection Error**
```bash
# Ensure MLflow service is running
docker compose up mlflow -d

# Check MLflow logs
docker compose logs mlflow

# Verify MLflow is accessible
curl http://localhost:5000
```

**AWS S3 Access Denied**
```bash
# Verify AWS credentials in .env file
# Ensure S3 bucket exists and has proper permissions
# Check IAM user has S3 read/write permissions
```

**Dataset Not Found in S3**
```bash
# Ensure uploader service has run successfully first
docker compose --profile manual run --rm uploader

# Check S3 bucket for dataset files:
# s3://your-bucket/datasets/ag_news/train.jsonl
# s3://your-bucket/datasets/ag_news/test.jsonl
```

**Out of Memory During Training**
```bash
# Reduce SAMPLE_SIZE in .env file
SAMPLE_SIZE=1000

# Or reduce batch size in trainer/train.py
per_device_train_batch_size=4
```

**Docker Build Failures**
```bash
# Clear Docker cache and rebuild
docker system prune -f
docker compose build --no-cache

# Check Docker has sufficient disk space
docker system df
```

### Service Debugging

```bash
# View service logs
docker compose logs uploader
docker compose logs trainer
docker compose logs mlflow

# Run services interactively for debugging
docker compose run --rm trainer bash
docker compose run --rm uploader bash

# Check container status
docker compose ps
```

### Monitoring Training Progress

- **MLflow UI**: http://localhost:5000 - View experiments, metrics, and artifacts
- **Training Logs**: Use `docker compose logs trainer` to monitor training output
- **S3 Console**: Check AWS S3 console for uploaded datasets and models
- **Resource Usage**: Monitor Docker stats with `docker stats`