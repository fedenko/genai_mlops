import os
import boto3
from datasets import load_dataset, Dataset
import pandas as pd
import tempfile
import json

from transformers import DistilBertTokenizerFast

from transformers import DistilBertForSequenceClassification

import mlflow
import mlflow.transformers
from transformers import Trainer, TrainingArguments

def upload_model_to_s3(model_path, s3_bucket):
    """Upload trained model to S3"""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'eu-central-1')
    )
    
    print(f"Uploading model to S3 bucket: {s3_bucket}")
    
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, model_path)
            s3_key = f"models/distilbert-agnews/{relative_path}"
            
            print(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
            s3_client.upload_file(local_path, s3_bucket, s3_key)
    
    print("Model upload to S3 completed!")

def load_dataset_from_s3(s3_bucket):
    """Load dataset from S3 that was uploaded by uploader service"""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'eu-central-1')
    )
    
    print(f"Loading dataset from S3 bucket: {s3_bucket}")
    
    # Create temporary directory for downloaded files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download train dataset
        train_file = os.path.join(temp_dir, 'train.jsonl')
        print("Downloading train dataset from S3...")
        s3_client.download_file(s3_bucket, 'datasets/ag_news/train.jsonl', train_file)
        
        # Download test dataset
        test_file = os.path.join(temp_dir, 'test.jsonl')
        print("Downloading test dataset from S3...")
        s3_client.download_file(s3_bucket, 'datasets/ag_news/test.jsonl', test_file)
        
        # Load datasets using pandas
        train_df = pd.read_json(train_file, lines=True)
        test_df = pd.read_json(test_file, lines=True)
        
        print(f"Loaded from S3 - Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        # Convert pandas DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Create DatasetDict-like structure
        dataset = {
            "train": train_dataset,
            "test": test_dataset
        }
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset from S3: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    print("Starting DistilBERT training on AG News dataset...")
    
    # Get configuration from environment
    sample_size = int(os.environ.get('SAMPLE_SIZE', 5000))
    s3_bucket = os.environ.get('S3_BUCKET')
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    
    # Load dataset from S3 (uploaded by uploader service)
    if not s3_bucket:
        raise ValueError("S3_BUCKET environment variable is required")
    
    dataset = load_dataset_from_s3(s3_bucket)
    
    # Select subset for training and testing based on sample_size
    train_samples = min(sample_size, len(dataset["train"]))
    test_samples = min(500, len(dataset["test"]))
    
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(train_samples))
    small_test_dataset = dataset["test"].shuffle(seed=42).select(range(test_samples))
    
    # Конвертуємо у pandas DataFrame для гарного відображення
    train_df = pd.DataFrame(small_train_dataset)
    test_df = pd.DataFrame(small_test_dataset)
    
    print("Train subset:")
    print(train_df.head())
    
    print("\nTest subset:")
    print(test_df.head())
    
    # Ініціалізуємо токенізатор для DistilBERT
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Функція токенізації: приймає приклад (рядок тексту) і повертає токенізовані дані
    def tokenize_function(example):
        return tokenizer(
            example["text"],           # текст для токенізації
            padding="max_length",      # додаємо паддінг до максимальної довжини послідовності
            truncation=True            # обрізаємо тексти довші за максимальну довжину
        )
    
    # Застосовуємо токенізацію до всього набору тренувальних даних
    tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
    
    # Аналогічно для тестового набору
    tokenized_test = small_test_dataset.map(tokenize_function, batched=True)
    
    # Завантажуємо попередньо натреновану модель DistilBERT для задачі класифікації
    # Вказуємо кількість класів — 4 (для AG News)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4
    )
    
    # Налаштування параметрів тренування
    training_args = TrainingArguments(
        output_dir="./results",                       # Каталог для збереження результатів
        learning_rate=2e-5,                          # Швидкість навчання
        per_device_train_batch_size=8,               # Розмір батчу для тренування
        per_device_eval_batch_size=8,                # Розмір батчу для валідації
        num_train_epochs=3,                          # Кількість епох
        weight_decay=0.01,                           # Зменшення ваг (регуляризація)
        logging_dir="./logs",                        # Каталог для логів
        logging_steps=10,                            # Кожні 10 кроків — логування
    )
    
    # Ініціалізуємо Trainer з моделлю та даними
    trainer = Trainer(
        model=model,                                 # Модель для тренування (DistilBERT)
        args=training_args,                          # Аргументи тренування
        train_dataset=tokenized_train,               # Тренувальна вибірка
        eval_dataset=tokenized_test,                 # Валідаційна вибірка
    )
    
    # Автоматичне логування параметрів, метрик і моделі за допомогою MLflow
    mlflow.transformers.autolog()
    
    # Починаємо сесію логування з MLflow
    with mlflow.start_run():
        trainer.train()                              # Запускаємо тренування
        
        # Оцінюємо модель на валідаційній вибірці та виводимо метрики
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)
        
        # Зберігаємо модель у папку "distilbert-agnews"
        trainer.save_model("distilbert-agnews")
        tokenizer.save_pretrained("distilbert-agnews")
        
        # Upload model to S3 if bucket is specified
        if s3_bucket:
            upload_model_to_s3("distilbert-agnews", s3_bucket)
        
        print("Training completed successfully!")

if __name__ == "__main__":
    main()