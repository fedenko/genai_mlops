"""
Hybrid NLP Pipeline for Ukrainian News
Combines text classification and summarization in a unified interface
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime

from classifier import UkrainianNewsClassifier, ClassicalMLClassifier, BERTClassifier
from summarizer import HybridSummarizer
from ..preprocessing.ukrainian_text import UkrainianTextProcessor

logger = logging.getLogger(__name__)


class UkrainianNewsHybridPipeline:
    """
    Unified pipeline for Ukrainian news processing
    Performs classification and conditional summarization
    """
    
    def __init__(self, 
                 classifier_model: str = 'bert',
                 summarization_model: str = 'ukr-models/uk-summarizer',
                 summarization_threshold: int = 500):
        """
        Initialize hybrid pipeline
        
        Args:
            classifier_model: Type of classifier ('bert', 'svm', 'random_forest')
            summarization_model: HuggingFace model for summarization
            summarization_threshold: Minimum text length for summarization
        """
        
        self.classifier_model = classifier_model
        self.summarization_model = summarization_model
        self.summarization_threshold = summarization_threshold
        
        # Initialize components
        self.text_processor = UkrainianTextProcessor()
        self.classifier = None
        self.summarizer = HybridSummarizer(ukrainian_model=summarization_model)
        
        # Pipeline metadata
        self.categories = None
        self.label_encoder = None
        self.is_trained = False
        
        # Performance tracking
        self.performance_stats = {
            'total_processed': 0,
            'classifications': 0,
            'summarizations': 0,
            'avg_processing_time': 0,
            'errors': 0
        }
    
    def initialize_classifier(self, num_labels: int):
        """Initialize classifier based on specified type"""
        
        if self.classifier_model == 'bert':
            self.classifier = BERTClassifier(
                model_name="google-bert/bert-base-multilingual-cased",
                num_labels=num_labels
            )
        elif self.classifier_model == 'svm':
            self.classifier = ClassicalMLClassifier(model_type='svm')
        elif self.classifier_model == 'random_forest':
            self.classifier = ClassicalMLClassifier(model_type='random_forest')
        else:
            raise ValueError(f"Unsupported classifier model: {self.classifier_model}")
        
        logger.info(f"Initialized {self.classifier_model} classifier for {num_labels} labels")
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
              text_column: str = 'combined_tokens', label_column: str = 'target_encoded') -> Dict[str, Any]:
        """
        Train the hybrid pipeline
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame (optional)
            text_column: Column containing preprocessed text
            label_column: Column containing encoded labels
            
        Returns:
            Dictionary with training results
        """
        
        logger.info("Starting hybrid pipeline training...")
        
        # Extract training data
        X_train = train_data[text_column].values
        y_train = train_data[label_column].values
        
        X_val = None
        y_val = None
        if val_data is not None:
            X_val = val_data[text_column].values
            y_val = val_data[label_column].values
        
        # Initialize classifier with correct number of labels
        num_labels = len(np.unique(y_train))
        self.initialize_classifier(num_labels)
        
        # Train classifier
        start_time = time.time()
        classifier_metrics = self.classifier.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        self.is_trained = True
        
        # Store categories for inference
        if 'target' in train_data.columns:
            self.categories = sorted(train_data['target'].unique())
        
        training_results = {
            'classifier_type': self.classifier_model,
            'classifier_metrics': classifier_metrics,
            'training_time': training_time,
            'num_labels': num_labels,
            'training_samples': len(X_train),
            'validation_samples': len(X_val) if X_val is not None else 0,
            'summarization_model': self.summarization_model,
            'summarization_threshold': self.summarization_threshold
        }
        
        logger.info(f"Hybrid pipeline training complete in {training_time:.2f}s")
        
        return training_results
    
    def process_single(self, title: str, text: str, 
                      include_summarization: bool = True,
                      summarization_method: str = 'auto') -> Dict[str, Any]:
        """
        Process a single news article
        
        Args:
            title: Article title
            text: Article content
            include_summarization: Whether to generate summary
            summarization_method: 'extractive', 'abstractive', or 'auto'
            
        Returns:
            Dictionary with classification and summarization results
        """
        
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess text
            combined_text = f"{title} {text}"
            clean_text = self.text_processor.clean_text(combined_text)
            tokens = ' '.join(self.text_processor.tokenize(clean_text))
            
            # Classification
            classification_start = time.time()
            prediction = self.classifier.predict([tokens])[0]
            
            # Get prediction probabilities for confidence
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba([tokens])[0]
                confidence = float(np.max(probabilities))
                class_probabilities = probabilities.tolist()
            else:
                confidence = 1.0
                class_probabilities = None
            
            classification_time = time.time() - classification_start
            
            # Map prediction to category name
            category = self.categories[prediction] if self.categories else str(prediction)
            
            # Summarization (conditional)
            summary_result = None
            summarization_time = 0
            
            if include_summarization and len(text) >= self.summarization_threshold:
                summarization_start = time.time()
                summary_result = self.summarizer.summarize(
                    text, 
                    method=summarization_method,
                    max_length=150
                )
                summarization_time = time.time() - summarization_start
                
                self.performance_stats['summarizations'] += 1
            
            total_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_processed'] += 1
            self.performance_stats['classifications'] += 1
            self.performance_stats['avg_processing_time'] = (
                (self.performance_stats['avg_processing_time'] * (self.performance_stats['total_processed'] - 1) + total_time) 
                / self.performance_stats['total_processed']
            )
            
            # Prepare result
            result = {
                'classification': {
                    'category': category,
                    'prediction': int(prediction),
                    'confidence': confidence,
                    'probabilities': class_probabilities,
                    'processing_time': classification_time
                },
                'summarization': summary_result,
                'metadata': {
                    'title_length': len(title),
                    'text_length': len(text),
                    'combined_length': len(combined_text),
                    'clean_text_length': len(clean_text),
                    'tokens_count': len(tokens.split()),
                    'total_processing_time': total_time,
                    'summarization_time': summarization_time,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            logger.debug(f"Processed article: {category} ({confidence:.3f}) in {total_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.performance_stats['errors'] += 1
            logger.error(f"Error processing article: {e}")
            
            return {
                'classification': {
                    'category': 'unknown',
                    'prediction': -1,
                    'confidence': 0.0,
                    'error': str(e)
                },
                'summarization': None,
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def process_batch(self, articles: List[Dict[str, str]], 
                     include_summarization: bool = True,
                     summarization_method: str = 'auto') -> List[Dict[str, Any]]:
        """
        Process multiple articles
        
        Args:
            articles: List of dictionaries with 'title' and 'text' keys
            include_summarization: Whether to generate summaries
            summarization_method: Summarization method to use
            
        Returns:
            List of processing results
        """
        
        logger.info(f"Processing batch of {len(articles)} articles")
        
        results = []
        for i, article in enumerate(articles):
            result = self.process_single(
                article['title'], 
                article['text'],
                include_summarization=include_summarization,
                summarization_method=summarization_method
            )
            
            result['batch_index'] = i
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(articles)} articles")
        
        logger.info(f"Batch processing complete: {len(results)} articles processed")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        
        stats = self.performance_stats.copy()
        
        # Add derived metrics
        if stats['total_processed'] > 0:
            stats['error_rate'] = stats['errors'] / stats['total_processed']
            stats['summarization_rate'] = stats['summarizations'] / stats['total_processed']
        
        stats['pipeline_config'] = {
            'classifier_model': self.classifier_model,
            'summarization_model': self.summarization_model,
            'summarization_threshold': self.summarization_threshold,
            'is_trained': self.is_trained
        }
        
        return stats
    
    def evaluate_on_test_set(self, test_data: pd.DataFrame,
                           text_column: str = 'combined_tokens',
                           label_column: str = 'target_encoded') -> Dict[str, Any]:
        """
        Evaluate pipeline on test set
        
        Args:
            test_data: Test DataFrame
            text_column: Text column name
            label_column: Label column name
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        logger.info(f"Evaluating on test set: {len(test_data)} samples")
        
        # Classification evaluation
        X_test = test_data[text_column].values
        y_test = test_data[label_column].values
        
        y_pred = self.classifier.predict(X_test)
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Summarization evaluation (if applicable)
        summarization_stats = {'evaluated': False}
        
        if 'text' in test_data.columns:
            long_texts = test_data[test_data['text'].str.len() >= self.summarization_threshold]
            
            if len(long_texts) > 0:
                # Sample a subset for summarization evaluation
                sample_size = min(100, len(long_texts))
                sample_texts = long_texts.sample(n=sample_size, random_state=42)
                
                summarization_times = []
                compression_ratios = []
                
                for _, row in sample_texts.iterrows():
                    start_time = time.time()
                    summary_result = self.summarizer.summarize(row['text'], method='auto')
                    summarization_times.append(time.time() - start_time)
                    compression_ratios.append(summary_result['compression_ratio'])
                
                summarization_stats = {
                    'evaluated': True,
                    'sample_size': sample_size,
                    'avg_summarization_time': np.mean(summarization_times),
                    'avg_compression_ratio': np.mean(compression_ratios),
                    'total_long_texts': len(long_texts)
                }
        
        evaluation_results = {
            'classification': {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'test_samples': len(test_data)
            },
            'summarization': summarization_stats,
            'pipeline_performance': self.get_performance_stats()
        }
        
        logger.info(f"Evaluation complete - Accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def save_pipeline(self, save_dir: Path):
        """Save the trained pipeline"""
        
        if not self.is_trained:
            raise ValueError("Pipeline not trained. Call train() first.")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        classifier_path = save_dir / f"classifier_{self.classifier_model}"
        self.classifier.save(classifier_path)
        
        # Save pipeline metadata
        metadata = {
            'classifier_model': self.classifier_model,
            'summarization_model': self.summarization_model,
            'summarization_threshold': self.summarization_threshold,
            'categories': self.categories,
            'performance_stats': self.performance_stats,
            'is_trained': self.is_trained,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(save_dir / 'pipeline_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Pipeline saved to {save_dir}")
    
    def load_pipeline(self, load_dir: Path):
        """Load a trained pipeline"""
        
        load_dir = Path(load_dir)
        
        # Load metadata
        with open(load_dir / 'pipeline_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.classifier_model = metadata['classifier_model']
        self.summarization_model = metadata['summarization_model']
        self.summarization_threshold = metadata['summarization_threshold']
        self.categories = metadata['categories']
        self.performance_stats = metadata.get('performance_stats', {})
        
        # Initialize and load classifier
        num_labels = len(self.categories) if self.categories else 5
        self.initialize_classifier(num_labels)
        
        classifier_path = load_dir / f"classifier_{self.classifier_model}"
        self.classifier.load(classifier_path)
        
        # Reinitialize summarizer
        self.summarizer = HybridSummarizer(ukrainian_model=self.summarization_model)
        
        self.is_trained = True
        
        logger.info(f"Pipeline loaded from {load_dir}")


def main():
    """Example usage of UkrainianNewsHybridPipeline"""
    
    # Sample data
    articles = [
        {
            'title': 'Політичні новини з України',
            'text': 'Сьогодні відбулася важлива зустріч політичних лідерів для обговорення майбутнього країни. ' * 10
        },
        {
            'title': 'Спортивні досягнення',
            'text': 'Українська збірна показала відмінні результати на міжнародних змаганнях.'
        }
    ]
    
    # Initialize pipeline
    pipeline = UkrainianNewsHybridPipeline(
        classifier_model='svm',
        summarization_threshold=200
    )
    
    # Process articles (this would require trained models in real usage)
    print("Hybrid pipeline example:")
    print(f"Configured for {pipeline.classifier_model} classification")
    print(f"Summarization threshold: {pipeline.summarization_threshold} characters")
    
    # Show pipeline configuration
    stats = pipeline.get_performance_stats()
    print(f"Pipeline configuration: {stats['pipeline_config']}")


if __name__ == "__main__":
    main()