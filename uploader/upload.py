import os
import boto3
from datasets import load_dataset
import pandas as pd
import json
import tempfile

def upload_dataset_to_s3():
    # Get environment variables
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION', 'eu-central-1')
    s3_bucket = os.environ.get('S3_BUCKET')
    
    if not all([aws_access_key_id, aws_secret_access_key, s3_bucket]):
        raise ValueError("Missing required environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET")
    
    print(f"Starting dataset upload to S3 bucket: {s3_bucket}")
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    
    try:
        # Load the dataset
        print("Loading wangrongsheng/ag_news dataset...")
        dataset = load_dataset("wangrongsheng/ag_news", download_mode="force_redownload")
        
        # Convert to pandas DataFrames
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
        
        print(f"Dataset loaded - Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        # Upload train dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            train_df.to_json(f.name, orient='records', lines=True)
            train_file = f.name
        
        print("Uploading train dataset to S3...")
        s3_client.upload_file(train_file, s3_bucket, 'datasets/ag_news/train.jsonl')
        os.unlink(train_file)
        
        # Upload test dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_df.to_json(f.name, orient='records', lines=True)
            test_file = f.name
        
        print("Uploading test dataset to S3...")
        s3_client.upload_file(test_file, s3_bucket, 'datasets/ag_news/test.jsonl')
        os.unlink(test_file)
        
        # Upload dataset info
        dataset_info = {
            "name": "wangrongsheng/ag_news",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "features": list(train_df.columns.tolist()),
            "labels": {
                "0": "World",
                "1": "Sports", 
                "2": "Business",
                "3": "Sci/Tech"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dataset_info, f, indent=2)
            info_file = f.name
        
        print("Uploading dataset info to S3...")
        s3_client.upload_file(info_file, s3_bucket, 'datasets/ag_news/info.json')
        os.unlink(info_file)
        
        print("Dataset upload completed successfully!")
        print(f"Files uploaded to s3://{s3_bucket}/datasets/ag_news/")
        
    except Exception as e:
        print(f"Error uploading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    upload_dataset_to_s3()
