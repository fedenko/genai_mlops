"""
Ukrainian News Text Classification Models
Supports multiple approaches: BERT, classical ML, and Ukrainian-specific models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle
import json

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Deep Learning
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, 
    pipeline as hf_pipeline
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class ClassicalMLClassifier:
    """Classical ML approaches for Ukrainian text classification"""
    
    def __init__(self, model_type: str = 'svm', max_features: int = 10000):
        self.model_type = model_type
        self.max_features = max_features
        
        # TF-IDF vectorizer for text features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words=None  # We handle Ukrainian stopwords in preprocessing
        )
        
        # Initialize model based on type
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.model)
        ])
        
        self.is_trained = False
        self.label_encoder = None
    
    def train(self, X_train: List[str], y_train: np.ndarray,
              X_val: Optional[List[str]] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the classical ML model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} classifier on {len(X_train)} samples")
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        metrics = {'train_accuracy': train_accuracy}
        
        if X_val is not None and y_val is not None:
            val_pred = self.pipeline.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            metrics['val_accuracy'] = val_accuracy
            
            logger.info(f"Training complete - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        else:
            logger.info(f"Training complete - Train Acc: {train_accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Check if model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            # For SVM with default kernel, return decision function as proxy
            decision = self.pipeline.decision_function(X)
            # Convert to pseudo-probabilities using softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
    
    def save(self, model_path: Path):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'model_type': self.model_type,
                'max_features': self.max_features
            }, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: Path):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.pipeline = data['pipeline']
        self.model_type = data['model_type']
        self.max_features = data['max_features']
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")


class BERTClassifier:
    """BERT-based classifier for Ukrainian text"""
    
    def __init__(self, model_name: str = "google-bert/bert-base-multilingual-cased",
                 num_labels: int = 5, max_length: int = 512):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        self.is_trained = False
        self.trainer = None
    
    def prepare_dataset(self, texts: List[str], labels: np.ndarray) -> Dataset:
        """Prepare dataset for BERT training"""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': labels.tolist()
        })
        
        # Tokenize
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, X_train: List[str], y_train: np.ndarray,
              X_val: Optional[List[str]] = None, y_val: Optional[np.ndarray] = None,
              output_dir: str = "models/bert_classifier",
              num_epochs: int = 3,
              batch_size: int = 16) -> Dict[str, float]:
        """
        Train BERT classifier
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training BERT classifier ({self.model_name}) on {len(X_train)} samples")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(X_train, y_train)
        
        eval_dataset = None
        if X_val is not None and y_val is not None:
            eval_dataset = self.prepare_dataset(X_val, y_val)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train model
        train_result = self.trainer.train()
        self.is_trained = True
        
        # Extract metrics
        metrics = {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second']
        }
        
        if eval_dataset:
            eval_result = self.trainer.evaluate()
            metrics.update({
                'eval_loss': eval_result['eval_loss'],
                'eval_runtime': eval_result['eval_runtime']
            })
        
        logger.info(f"BERT training complete - Train Loss: {metrics['train_loss']:.4f}")
        
        return metrics
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Make predictions using trained BERT model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create prediction pipeline
        classifier = hf_pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=False
        )
        
        # Get predictions
        predictions = classifier(X)
        
        # Extract predicted labels
        pred_labels = [int(pred['label'].split('_')[-1]) for pred in predictions]
        
        return np.array(pred_labels)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        classifier = hf_pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            return_all_scores=True
        )
        
        predictions = classifier(X)
        
        # Convert to probability matrix
        probabilities = []
        for pred_scores in predictions:
            scores = [score['score'] for score in sorted(pred_scores, key=lambda x: int(x['label'].split('_')[-1]))]
            probabilities.append(scores)
        
        return np.array(probabilities)
    
    def save(self, model_path: Path):
        """Save trained BERT model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length
        }
        
        with open(model_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"BERT model saved to {model_path}")
    
    def load(self, model_path: Path):
        """Load trained BERT model"""
        
        # Load metadata
        with open(model_path / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.num_labels = metadata['num_labels']
        self.max_length = metadata['max_length']
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.is_trained = True
        
        logger.info(f"BERT model loaded from {model_path}")


class UkrainianNewsClassifier:
    """Unified interface for Ukrainian news classification"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.label_encoder = None
        self.categories = None
    
    def add_model(self, name: str, model: Any):
        """Add a model to the ensemble"""
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def train_all_models(self, X_train: List[str], y_train: np.ndarray,
                        X_val: Optional[List[str]] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """Train all added models and compare performance"""
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            
            try:
                metrics = model.train(X_train, y_train, X_val, y_val)
                results[name] = metrics
                
                # Evaluate on validation set
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_accuracy = accuracy_score(y_val, val_pred)
                    results[name]['final_val_accuracy'] = val_accuracy
                
            except Exception as e:
                logger.error(f"Failed to train model {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Find best model based on validation accuracy
        if X_val is not None and y_val is not None:
            best_score = 0
            for name, metrics in results.items():
                if 'final_val_accuracy' in metrics and metrics['final_val_accuracy'] > best_score:
                    best_score = metrics['final_val_accuracy']
                    self.best_model = name
            
            logger.info(f"Best model: {self.best_model} (accuracy: {best_score:.4f})")
        
        return results
    
    def predict(self, X: List[str], model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using specified model or best model"""
        
        model_name = model_name or self.best_model
        if model_name is None:
            raise ValueError("No model specified and no best model identified")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def get_model_comparison(self, X_test: List[str], y_test: np.ndarray) -> pd.DataFrame:
        """Compare all trained models on test set"""
        
        results = []
        
        for name, model in self.models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                try:
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    results.append({
                        'model': name,
                        'accuracy': accuracy,
                        'predictions': len(y_pred)
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate model {name}: {e}")
        
        return pd.DataFrame(results)


def main():
    """Example usage of Ukrainian news classifiers"""
    
    # Sample data (replace with actual Ukrainian news data)
    X_train = ["Це новина про політику", "Спортивна новина про футбол", "Бізнес новина"]
    y_train = np.array([0, 1, 2])
    
    # Initialize classical ML classifier
    classical_model = ClassicalMLClassifier(model_type='svm')
    
    # Initialize BERT classifier
    bert_model = BERTClassifier(num_labels=3)
    
    # Create unified classifier
    classifier = UkrainianNewsClassifier()
    classifier.add_model('svm', classical_model)
    classifier.add_model('bert', bert_model)
    
    # Train all models
    results = classifier.train_all_models(X_train, y_train)
    
    print("Training results:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: {metrics}")


if __name__ == "__main__":
    main()