"""
Main Training Script for Ukrainian News Hybrid Pipeline
Orchestrates data loading, preprocessing, model training, and evaluation
"""

import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, List, Optional, Any
import warnings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.data_pipeline import UkrainianNewsDataPipeline
from models.classifier import ClassicalMLClassifier, BERTClassifier
from models.summarizer import HybridSummarizer
from models.hybrid_pipeline import UkrainianNewsHybridPipeline
from training.mlflow_tracker import UkrainianNLPExperimentTracker

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UkrainianNewsTrainer:
    """Main trainer class for Ukrainian news pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration dictionary
        """
        
        self.config = config
        
        # Initialize components
        self.data_pipeline = UkrainianNewsDataPipeline(
            data_dir=config.get('data_dir', 'data')
        )
        
        self.experiment_tracker = UkrainianNLPExperimentTracker(
            experiment_name=config.get('experiment_name', 'ukrainian_news_pipeline'),
            tracking_uri=config.get('mlflow_tracking_uri')
        )
        
        # Data storage
        self.datasets = {}
        self.data_analysis = {}
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        
    def load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess the Ukrainian news dataset"""
        
        logger.info("Starting data loading and preprocessing...")
        
        # Load dataset
        datasets = self.data_pipeline.load_dataset(
            dataset_name=self.config.get('dataset_name', 'FIdo-AI/ua-news'),
            cache_dir=self.config.get('cache_dir')
        )
        
        # Analyze dataset
        self.data_analysis = self.data_pipeline.analyze_dataset()
        
        logger.info("Dataset analysis:")
        for key, value in self.data_analysis.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Preprocess data
        self.datasets = self.data_pipeline.preprocess_data(
            create_validation=True,
            val_size=self.config.get('validation_size', 0.2)
        )
        
        logger.info(f"Data preprocessing complete:")
        for split, df in self.datasets.items():
            logger.info(f"  {split}: {len(df)} samples")
        
        return self.datasets
    
    def train_classification_models(self) -> Dict[str, Any]:
        """Train multiple classification models and compare them"""
        
        logger.info("Starting classification model training...")
        
        # Prepare classification data
        classification_data = self.data_pipeline.prepare_classification_data(
            text_column=self.config.get('text_column', 'combined_tokens')
        )
        
        X_train, y_train = classification_data['train']
        X_val, y_val = classification_data['validation']
        X_test, y_test = classification_data['test']
        
        num_labels = len(np.unique(y_train))
        category_names = self.data_pipeline.get_category_names()
        
        logger.info(f"Training classification models for {num_labels} categories: {category_names}")
        
        classification_results = {}
        
        # Train models specified in config
        models_to_train = self.config.get('classification_models', ['svm', 'bert'])
        
        for model_name in models_to_train:
            logger.info(f"Training {model_name} classifier...")
            
            try:
                # Initialize model
                if model_name == 'svm':
                    model = ClassicalMLClassifier(model_type='svm')
                elif model_name == 'random_forest':
                    model = ClassicalMLClassifier(model_type='random_forest')
                elif model_name == 'logistic':
                    model = ClassicalMLClassifier(model_type='logistic')
                elif model_name == 'bert':
                    model = BERTClassifier(
                        model_name=self.config.get('bert_model', 'google-bert/bert-base-multilingual-cased'),
                        num_labels=num_labels,
                        max_length=self.config.get('max_length', 512)
                    )
                else:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
                
                # Train model
                if model_name == 'bert':
                    training_metrics = model.train(
                        X_train, y_train, X_val, y_val,
                        num_epochs=self.config.get('bert_epochs', 3),
                        batch_size=self.config.get('bert_batch_size', 16)
                    )
                else:
                    training_metrics = model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                
                from sklearn.metrics import accuracy_score, classification_report
                test_accuracy = accuracy_score(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Store results
                classification_results[model_name] = {
                    'model': model,
                    'training_metrics': training_metrics,
                    'test_accuracy': test_accuracy,
                    'classification_report': class_report,
                    'predictions': y_pred
                }
                
                logger.info(f"{model_name} training complete - Test Accuracy: {test_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                classification_results[model_name] = {'error': str(e)}
        
        self.training_results['classification'] = classification_results
        
        return classification_results
    
    def evaluate_summarization(self) -> Dict[str, Any]:
        """Evaluate summarization models"""
        
        logger.info("Starting summarization evaluation...")
        
        # Get summarization data (long texts)
        summarization_data = self.data_pipeline.prepare_summarization_data(
            min_length=self.config.get('summarization_min_length', 500)
        )
        
        # Initialize summarizer
        summarizer = HybridSummarizer(
            ukrainian_model=self.config.get('summarization_model', 'ukr-models/uk-summarizer')
        )
        
        summarization_results = {}
        
        # Evaluate on test set sample
        test_df = summarization_data['test']
        sample_size = min(self.config.get('summarization_eval_samples', 100), len(test_df))
        test_sample = test_df.sample(n=sample_size, random_state=42)
        
        logger.info(f"Evaluating summarization on {sample_size} samples...")
        
        # Test different methods
        methods = ['extractive', 'abstractive', 'auto']
        
        for method in methods:
            logger.info(f"Testing {method} summarization...")
            
            method_results = {
                'summaries': [],
                'compression_ratios': [],
                'processing_times': []
            }
            
            for _, row in test_sample.iterrows():
                try:
                    import time
                    start_time = time.time()
                    
                    summary_result = summarizer.summarize(
                        row['text'], 
                        method=method,
                        max_length=self.config.get('max_summary_length', 150)
                    )
                    
                    processing_time = time.time() - start_time
                    
                    method_results['summaries'].append({
                        'original': row['text'],
                        'summary': summary_result['summary'],
                        'method': summary_result['method'],
                        'compression_ratio': summary_result['compression_ratio']
                    })
                    
                    method_results['compression_ratios'].append(summary_result['compression_ratio'])
                    method_results['processing_times'].append(processing_time)
                    
                except Exception as e:
                    logger.warning(f"Summarization failed for {method}: {e}")
            
            # Calculate aggregate metrics
            if method_results['compression_ratios']:
                method_results['avg_compression_ratio'] = np.mean(method_results['compression_ratios'])
                method_results['avg_processing_time'] = np.mean(method_results['processing_times'])
                method_results['successful_summaries'] = len(method_results['compression_ratios'])
            
            summarization_results[method] = method_results
            
            logger.info(f"{method} summarization complete - "
                       f"Avg compression: {method_results.get('avg_compression_ratio', 0):.3f}, "
                       f"Avg time: {method_results.get('avg_processing_time', 0):.3f}s")
        
        self.training_results['summarization'] = summarization_results
        
        return summarization_results
    
    def train_hybrid_pipeline(self) -> Dict[str, Any]:
        """Train the complete hybrid pipeline"""
        
        logger.info("Starting hybrid pipeline training...")
        
        # Get best classification model
        classification_results = self.training_results.get('classification', {})
        
        best_model_name = 'bert'  # Default
        best_accuracy = 0
        
        for model_name, results in classification_results.items():
            if 'test_accuracy' in results and results['test_accuracy'] > best_accuracy:
                best_accuracy = results['test_accuracy']
                best_model_name = model_name
        
        logger.info(f"Using best classification model: {best_model_name} (accuracy: {best_accuracy:.4f})")
        
        # Initialize hybrid pipeline
        pipeline = UkrainianNewsHybridPipeline(
            classifier_model=best_model_name,
            summarization_model=self.config.get('summarization_model', 'ukr-models/uk-summarizer'),
            summarization_threshold=self.config.get('summarization_threshold', 500)
        )
        
        # Train pipeline
        training_results = pipeline.train(
            train_data=self.datasets['train'],
            val_data=self.datasets['validation'],
            text_column=self.config.get('text_column', 'combined_tokens'),
            label_column='target_encoded'
        )
        
        # Evaluate pipeline
        evaluation_results = pipeline.evaluate_on_test_set(
            test_data=self.datasets['test'],
            text_column=self.config.get('text_column', 'combined_tokens'),
            label_column='target_encoded'
        )
        
        # Get performance stats
        performance_stats = pipeline.get_performance_stats()
        
        # Save pipeline
        if self.config.get('save_pipeline', True):
            save_dir = Path(self.config.get('models_dir', 'models')) / 'hybrid_pipeline'
            pipeline.save_pipeline(save_dir)
            logger.info(f"Hybrid pipeline saved to {save_dir}")
        
        hybrid_results = {
            'pipeline': pipeline,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'performance_stats': performance_stats
        }
        
        self.training_results['hybrid_pipeline'] = hybrid_results
        
        return hybrid_results
    
    def log_experiments_to_mlflow(self):
        """Log all experiments to MLflow"""
        
        logger.info("Logging experiments to MLflow...")
        
        # Start MLflow run
        run_id = self.experiment_tracker.start_run(
            run_name=self.config.get('run_name', f"ukrainian_news_training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"),
            description=self.config.get('description', "Ukrainian news pipeline training"),
            tags=self.config.get('tags', {})
        )
        
        # Log data information
        self.experiment_tracker.log_data_info(self.data_analysis)
        
        # Log classification experiments
        if 'classification' in self.training_results:
            for model_name, results in self.training_results['classification'].items():
                if 'error' not in results:
                    model_params = {
                        'model_type': model_name,
                        'num_labels': len(self.data_pipeline.get_category_names())
                    }
                    
                    self.experiment_tracker.log_classification_experiment(
                        model_name=model_name,
                        model_params=model_params,
                        training_metrics=results.get('training_metrics', {}),
                        test_metrics={'accuracy': results.get('test_accuracy', 0)},
                        model_artifact=results.get('model')
                    )
        
        # Log summarization experiments
        if 'summarization' in self.training_results:
            for method, results in self.training_results['summarization'].items():
                if 'avg_compression_ratio' in results:
                    model_params = {
                        'method': method,
                        'max_length': self.config.get('max_summary_length', 150)
                    }
                    
                    evaluation_metrics = {
                        'avg_compression_ratio': results['avg_compression_ratio'],
                        'avg_processing_time': results['avg_processing_time'],
                        'successful_summaries': results['successful_summaries']
                    }
                    
                    self.experiment_tracker.log_summarization_experiment(
                        model_name=self.config.get('summarization_model', 'ukr-models/uk-summarizer'),
                        summarization_method=method,
                        model_params=model_params,
                        evaluation_metrics=evaluation_metrics,
                        sample_summaries=results['summaries'][:5]  # Log first 5 samples
                    )
        
        # Log hybrid pipeline experiment
        if 'hybrid_pipeline' in self.training_results:
            hybrid_results = self.training_results['hybrid_pipeline']
            
            pipeline_config = {
                'classifier_model': self.config.get('classification_models', ['bert'])[0],
                'summarization_model': self.config.get('summarization_model'),
                'summarization_threshold': self.config.get('summarization_threshold', 500)
            }
            
            self.experiment_tracker.log_hybrid_pipeline_experiment(
                pipeline_config=pipeline_config,
                training_results=hybrid_results['training_results'],
                evaluation_results=hybrid_results['evaluation_results'],
                performance_stats=hybrid_results['performance_stats']
            )
        
        # End MLflow run
        self.experiment_tracker.end_run()
        
        logger.info(f"Experiments logged to MLflow. Run ID: {run_id}")
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        
        logger.info("Starting complete Ukrainian news pipeline training...")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2, default=str)}")
        
        try:
            # 1. Load and preprocess data
            self.load_and_preprocess_data()
            
            # 2. Train classification models
            self.train_classification_models()
            
            # 3. Evaluate summarization
            self.evaluate_summarization()
            
            # 4. Train hybrid pipeline
            self.train_hybrid_pipeline()
            
            # 5. Log to MLflow
            self.log_experiments_to_mlflow()
            
            logger.info("Training pipeline completed successfully!")
            
            # Print summary
            self.print_training_summary()
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def print_training_summary(self):
        """Print a summary of training results"""
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        # Data summary
        print(f"\nDataset: {self.data_analysis.get('train_samples', 0)} train, "
              f"{self.data_analysis.get('test_samples', 0)} test samples")
        print(f"Categories: {len(self.data_analysis.get('categories', []))}")
        
        # Classification results
        if 'classification' in self.training_results:
            print("\nClassification Results:")
            for model_name, results in self.training_results['classification'].items():
                if 'test_accuracy' in results:
                    print(f"  {model_name}: {results['test_accuracy']:.4f} accuracy")
        
        # Summarization results
        if 'summarization' in self.training_results:
            print("\nSummarization Results:")
            for method, results in self.training_results['summarization'].items():
                if 'avg_compression_ratio' in results:
                    print(f"  {method}: {results['avg_compression_ratio']:.3f} compression, "
                          f"{results['avg_processing_time']:.3f}s avg time")
        
        # Hybrid pipeline results
        if 'hybrid_pipeline' in self.training_results:
            hybrid_results = self.training_results['hybrid_pipeline']
            eval_results = hybrid_results['evaluation_results']
            
            print("\nHybrid Pipeline Results:")
            if 'classification' in eval_results:
                print(f"  Test Accuracy: {eval_results['classification']['accuracy']:.4f}")
            
            if 'summarization' in eval_results and eval_results['summarization']['evaluated']:
                summ_results = eval_results['summarization']
                print(f"  Avg Summarization Time: {summ_results['avg_summarization_time']:.3f}s")
                print(f"  Avg Compression Ratio: {summ_results['avg_compression_ratio']:.3f}")
        
        print("\n" + "="*80)


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration"""
    
    return {
        # Data settings
        'dataset_name': 'FIdo-AI/ua-news',
        'data_dir': 'data',
        'cache_dir': None,
        'validation_size': 0.2,
        'text_column': 'combined_tokens',
        
        # Classification settings
        'classification_models': ['svm', 'bert'],
        'bert_model': 'google-bert/bert-base-multilingual-cased',
        'bert_epochs': 3,
        'bert_batch_size': 16,
        'max_length': 512,
        
        # Summarization settings
        'summarization_model': 'ukr-models/uk-summarizer',
        'summarization_min_length': 500,
        'summarization_threshold': 500,
        'max_summary_length': 150,
        'summarization_eval_samples': 100,
        
        # Pipeline settings
        'save_pipeline': True,
        'models_dir': 'models',
        
        # MLflow settings
        'experiment_name': 'ukrainian_news_pipeline',
        'mlflow_tracking_uri': None,
        'run_name': None,
        'description': 'Ukrainian news classification and summarization pipeline',
        'tags': {'project': 'ukrainian_nlp', 'version': '1.0'}
    }


def main():
    """Main function for training script"""
    
    parser = argparse.ArgumentParser(description='Train Ukrainian News Hybrid Pipeline')
    
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--experiment-name', type=str, help='MLflow experiment name')
    parser.add_argument('--models', nargs='+', help='Classification models to train',
                       choices=['svm', 'random_forest', 'logistic', 'bert'])
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for BERT training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for BERT training')
    parser.add_argument('--no-save', action='store_true', help='Do not save trained models')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    
    if args.models:
        config['classification_models'] = args.models
    
    if args.epochs:
        config['bert_epochs'] = args.epochs
    
    if args.batch_size:
        config['bert_batch_size'] = args.batch_size
    
    if args.no_save:
        config['save_pipeline'] = False
    
    # Initialize trainer and run training
    trainer = UkrainianNewsTrainer(config)
    trainer.run_complete_training()


if __name__ == "__main__":
    main()