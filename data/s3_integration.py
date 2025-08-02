"""
S3 Integration for Ukrainian News Dataset
Handles uploading and downloading of processed data to/from AWS S3
"""

import boto3
import pandas as pd
import pickle
import json
from pathlib import Path
from datasets import load_dataset
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class S3DataManager:
    """Manages Ukrainian news data upload/download to/from S3"""
    
    def __init__(self, bucket_name: str, aws_region: str = "us-west-2"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
    def upload_dataset_to_s3(self, dataset_name: str = "FIdo-AI/ua-news") -> Dict[str, str]:
        """
        Load Ukrainian news dataset and upload to S3
        
        Returns:
            Dict with S3 paths of uploaded files
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load the dataset
        dataset = load_dataset(dataset_name)
        
        # Convert to pandas for easier processing
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        # Save locally first
        train_path = Path("data/raw/train.parquet")
        test_path = Path("data/raw/test.parquet")
        
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        
        # Upload to S3
        train_s3_key = "data/raw/ua_news_train.parquet"
        test_s3_key = "data/raw/ua_news_test.parquet"
        
        self.s3_client.upload_file(str(train_path), self.bucket_name, train_s3_key)
        self.s3_client.upload_file(str(test_path), self.bucket_name, test_s3_key)
        
        # Create metadata
        metadata = {
            "dataset_name": dataset_name,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "columns": list(train_df.columns),
            "categories": train_df['target'].unique().tolist() if 'target' in train_df.columns else [],
            "train_s3_path": f"s3://{self.bucket_name}/{train_s3_key}",
            "test_s3_path": f"s3://{self.bucket_name}/{test_s3_key}"
        }
        
        # Upload metadata
        metadata_key = "data/metadata/ua_news_metadata.json"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=metadata_key,
            Body=json.dumps(metadata, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Dataset uploaded successfully. Train: {len(train_df)}, Test: {len(test_df)}")
        
        return {
            "train_s3_path": f"s3://{self.bucket_name}/{train_s3_key}",
            "test_s3_path": f"s3://{self.bucket_name}/{test_s3_key}",
            "metadata_s3_path": f"s3://{self.bucket_name}/{metadata_key}"
        }
    
    def download_dataset_from_s3(self, local_dir: str = "data/raw") -> Dict[str, Path]:
        """Download dataset from S3 to local directory"""
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download files
        train_local = local_dir / "train.parquet"
        test_local = local_dir / "test.parquet"
        metadata_local = local_dir / "metadata.json"
        
        self.s3_client.download_file(
            self.bucket_name, 
            "data/raw/ua_news_train.parquet", 
            str(train_local)
        )
        
        self.s3_client.download_file(
            self.bucket_name, 
            "data/raw/ua_news_test.parquet", 
            str(test_local)
        )
        
        self.s3_client.download_file(
            self.bucket_name, 
            "data/metadata/ua_news_metadata.json", 
            str(metadata_local)
        )
        
        return {
            "train_path": train_local,
            "test_path": test_local,
            "metadata_path": metadata_local
        }
    
    def upload_processed_features(self, features_dict: Dict[str, Any], 
                                 feature_set_name: str) -> str:
        """Upload processed features to S3"""
        
        # Serialize features
        features_key = f"data/processed/{feature_set_name}_features.pkl"
        
        # Save locally first
        local_path = Path(f"data/processed/{feature_set_name}_features.pkl")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        # Upload to S3
        self.s3_client.upload_file(str(local_path), self.bucket_name, features_key)
        
        return f"s3://{self.bucket_name}/{features_key}"
    
    def upload_model_artifacts(self, model_path: Path, model_name: str, 
                             version: str) -> str:
        """Upload trained model artifacts to S3"""
        
        s3_key = f"models/{model_name}/v{version}/{model_path.name}"
        
        self.s3_client.upload_file(str(model_path), self.bucket_name, s3_key)
        
        return f"s3://{self.bucket_name}/{s3_key}"


def main():
    """Example usage of S3DataManager"""
    
    # Initialize S3 manager (replace with your bucket)
    s3_manager = S3DataManager("your-mlops-bucket")
    
    # Upload dataset to S3
    s3_paths = s3_manager.upload_dataset_to_s3()
    print("Uploaded dataset to S3:")
    for key, path in s3_paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()