"""
Data Pipeline for Ukrainian News Dataset
Loads, preprocesses, and prepares data for ML models
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ukrainian_text import UkrainianTextProcessor

logger = logging.getLogger(__name__)


class UkrainianNewsDataPipeline:
    """Complete data pipeline for Ukrainian news classification and summarization"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_processor = UkrainianTextProcessor()
        self.label_encoder = LabelEncoder()
        
        # Dataset metadata
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.categories = None
        
    def load_dataset(self, dataset_name: str = "FIdo-AI/ua-news", 
                    cache_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load Ukrainian news dataset from HuggingFace
        
        Args:
            dataset_name: HuggingFace dataset identifier
            cache_dir: Directory to cache dataset
            
        Returns:
            Dictionary with train/test DataFrames
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
            
            # Convert to pandas
            self.train_df = dataset['train'].to_pandas()
            self.test_df = dataset['test'].to_pandas()
            
            logger.info(f"Loaded dataset - Train: {len(self.train_df)}, Test: {len(self.test_df)}")
            
            # Analyze categories
            if 'target' in self.train_df.columns:
                self.categories = sorted(self.train_df['target'].unique())
                logger.info(f"Found {len(self.categories)} categories: {self.categories}")
            
            # Save raw data locally
            raw_dir = self.data_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            
            self.train_df.to_parquet(raw_dir / "train_raw.parquet")
            self.test_df.to_parquet(raw_dir / "test_raw.parquet")
            
            return {
                'train': self.train_df,
                'test': self.test_df
            }
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def analyze_dataset(self) -> Dict[str, any]:
        """
        Analyze the loaded dataset and return statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.train_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        analysis = {}
        
        # Basic statistics
        analysis['train_samples'] = len(self.train_df)
        analysis['test_samples'] = len(self.test_df)
        analysis['columns'] = list(self.train_df.columns)
        
        # Text length analysis
        if 'text' in self.train_df.columns:
            text_lengths = self.train_df['text'].str.len()
            analysis['text_length_stats'] = {
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'std': float(text_lengths.std())
            }
        
        # Category analysis
        if 'target' in self.train_df.columns:
            category_counts = self.train_df['target'].value_counts()
            analysis['category_distribution'] = category_counts.to_dict()
            analysis['categories'] = self.categories
        
        # Missing values
        analysis['missing_values'] = self.train_df.isnull().sum().to_dict()
        
        # Sample sizes for summarization (long texts)
        if 'text' in self.train_df.columns:
            long_texts = self.train_df[self.train_df['text'].str.len() > 500]
            analysis['long_text_samples'] = len(long_texts)
            analysis['summarization_candidates'] = float(len(long_texts) / len(self.train_df))
        
        logger.info(f"Dataset analysis complete: {analysis['train_samples']} train samples, {analysis['test_samples']} test samples")
        
        return analysis
    
    def preprocess_data(self, create_validation: bool = True, 
                       val_size: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Preprocess the dataset for ML training
        
        Args:
            create_validation: Whether to create validation split
            val_size: Validation set size (fraction of training data)
            
        Returns:
            Dictionary with preprocessed DataFrames
        """
        if self.train_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Starting data preprocessing...")
        
        # Preprocess text using Ukrainian processor
        train_processed = self.text_processor.preprocess_dataset(self.train_df.copy())
        test_processed = self.text_processor.preprocess_dataset(self.test_df.copy())
        
        # Encode labels
        if 'target' in train_processed.columns:
            # Fit encoder on training data
            train_processed['target_encoded'] = self.label_encoder.fit_transform(train_processed['target'])
            test_processed['target_encoded'] = self.label_encoder.transform(test_processed['target'])
            
            logger.info(f"Encoded {len(self.label_encoder.classes_)} categories: {list(self.label_encoder.classes_)}")
        
        # Create validation split
        if create_validation and 'target_encoded' in train_processed.columns:
            train_data, val_data = train_test_split(
                train_processed, 
                test_size=val_size, 
                stratify=train_processed['target_encoded'],
                random_state=42
            )
            
            self.train_df = train_data
            self.val_df = val_data
            self.test_df = test_processed
            
            logger.info(f"Created validation split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_processed)}")
        else:
            self.train_df = train_processed
            self.test_df = test_processed
        
        # Save processed data
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        self.train_df.to_parquet(processed_dir / "train_processed.parquet")
        self.test_df.to_parquet(processed_dir / "test_processed.parquet")
        
        if self.val_df is not None:
            self.val_df.to_parquet(processed_dir / "val_processed.parquet")
        
        # Save label encoder
        import pickle
        with open(processed_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        result = {
            'train': self.train_df,
            'test': self.test_df
        }
        
        if self.val_df is not None:
            result['validation'] = self.val_df
        
        logger.info("Data preprocessing complete")
        
        return result
    
    def prepare_classification_data(self, text_column: str = 'combined_tokens') -> Dict[str, Tuple]:
        """
        Prepare data specifically for classification models
        
        Args:
            text_column: Column to use as input text
            
        Returns:
            Dictionary with (X, y) tuples for each split
        """
        if self.train_df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        result = {}
        
        # Training data
        X_train = self.train_df[text_column].values
        y_train = self.train_df['target_encoded'].values
        result['train'] = (X_train, y_train)
        
        # Validation data
        if self.val_df is not None:
            X_val = self.val_df[text_column].values
            y_val = self.val_df['target_encoded'].values
            result['validation'] = (X_val, y_val)
        
        # Test data
        X_test = self.test_df[text_column].values
        y_test = self.test_df['target_encoded'].values
        result['test'] = (X_test, y_test)
        
        logger.info(f"Prepared classification data using column: {text_column}")
        
        return result
    
    def prepare_summarization_data(self, min_length: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for text summarization (filter long texts)
        
        Args:
            min_length: Minimum text length for summarization candidates
            
        Returns:
            Dictionary with filtered DataFrames for summarization
        """
        if self.train_df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Filter texts that are long enough for summarization
        train_long = self.train_df[self.train_df['text_text_length'] >= min_length].copy()
        test_long = self.test_df[self.test_df['text_text_length'] >= min_length].copy()
        
        result = {
            'train': train_long,
            'test': test_long
        }
        
        if self.val_df is not None:
            val_long = self.val_df[self.val_df['text_text_length'] >= min_length].copy()
            result['validation'] = val_long
        
        logger.info(f"Prepared summarization data: Train {len(train_long)}, Test {len(test_long)} (min_length={min_length})")
        
        return result
    
    def get_category_names(self) -> List[str]:
        """Get the list of category names"""
        if self.label_encoder is None:
            raise ValueError("Labels not encoded. Call preprocess_data() first.")
        
        return list(self.label_encoder.classes_)


def main():
    """Example usage of UkrainianNewsDataPipeline"""
    
    # Initialize pipeline
    pipeline = UkrainianNewsDataPipeline()
    
    # Load dataset
    datasets = pipeline.load_dataset()
    
    # Analyze dataset
    analysis = pipeline.analyze_dataset()
    print("Dataset Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Preprocess data
    processed_data = pipeline.preprocess_data()
    
    # Prepare for classification
    classification_data = pipeline.prepare_classification_data()
    print(f"\nClassification data shapes:")
    for split, (X, y) in classification_data.items():
        print(f"  {split}: X={X.shape}, y={y.shape}")
    
    # Prepare for summarization
    summarization_data = pipeline.prepare_summarization_data()
    print(f"\nSummarization data:")
    for split, df in summarization_data.items():
        print(f"  {split}: {len(df)} samples")


if __name__ == "__main__":
    main()