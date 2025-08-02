"""
MLflow Experiment Tracking for Ukrainian News Pipeline
Tracks experiments, metrics, and artifacts for classification and summarization
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

from sklearn.metrics import classification_report, confusion_matrix
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class UkrainianNLPExperimentTracker:
    """MLflow experiment tracker for Ukrainian NLP pipeline"""
    
    def __init__(self, experiment_name: str = "ukrainian_news_pipeline",
                 tracking_uri: Optional[str] = None):
        """
        Initialize MLflow experiment tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (optional)
        """
        
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        logger.info(f"MLflow experiment tracker initialized: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, 
                  description: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name for the run
            description: Description of the experiment
            tags: Additional tags for the run
            
        Returns:
            Run ID
        """
        
        # Prepare tags
        run_tags = {
            'project': 'ukrainian_news_pipeline',
            'timestamp': datetime.now().isoformat()
        }
        
        if tags:
            run_tags.update(tags)
        
        if description:
            run_tags['description'] = description
        
        # Start run
        run = mlflow.start_run(run_name=run_name, tags=run_tags)
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        
        return run.info.run_id
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """Log dataset information"""
        
        # Log dataset parameters
        mlflow.log_param("dataset_name", data_info.get('dataset_name', 'unknown'))
        mlflow.log_param("train_samples", data_info.get('train_samples', 0))
        mlflow.log_param("test_samples", data_info.get('test_samples', 0))
        mlflow.log_param("validation_samples", data_info.get('validation_samples', 0))
        mlflow.log_param("num_categories", len(data_info.get('categories', [])))
        
        # Log text statistics
        if 'text_length_stats' in data_info:
            text_stats = data_info['text_length_stats']
            mlflow.log_metric("avg_text_length", text_stats.get('mean', 0))
            mlflow.log_metric("median_text_length", text_stats.get('median', 0))
            mlflow.log_metric("max_text_length", text_stats.get('max', 0))
            mlflow.log_metric("text_length_std", text_stats.get('std', 0))
        
        # Log category distribution
        if 'category_distribution' in data_info:
            category_dist = data_info['category_distribution']
            for category, count in category_dist.items():
                mlflow.log_metric(f"category_{category}_count", count)
        
        # Log summarization candidates
        if 'summarization_candidates' in data_info:
            mlflow.log_metric("summarization_candidate_ratio", data_info['summarization_candidates'])
        
        logger.info("Data information logged to MLflow")
    
    def log_classification_experiment(self, 
                                    model_name: str,
                                    model_params: Dict[str, Any],
                                    training_metrics: Dict[str, float],
                                    validation_metrics: Optional[Dict[str, float]] = None,
                                    test_metrics: Optional[Dict[str, float]] = None,
                                    model_artifact: Optional[Any] = None,
                                    artifacts_dir: Optional[Path] = None):
        """
        Log classification experiment results
        
        Args:
            model_name: Name of the classification model
            model_params: Model hyperparameters
            training_metrics: Training metrics
            validation_metrics: Validation metrics (optional)
            test_metrics: Test metrics (optional)
            model_artifact: Trained model object (optional)
            artifacts_dir: Directory to save artifacts (optional)
        """
        
        # Log model parameters
        mlflow.log_param("model_type", "classification")
        mlflow.log_param("model_name", model_name)
        
        for param, value in model_params.items():
            mlflow.log_param(f"model_{param}", value)
        
        # Log training metrics
        for metric, value in training_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        
        # Log validation metrics
        if validation_metrics:
            for metric, value in validation_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
        
        # Log test metrics
        if test_metrics:
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
        
        # Log model artifact
        if model_artifact is not None:
            if hasattr(model_artifact, 'pipeline'):
                # Sklearn-based model
                mlflow.sklearn.log_model(model_artifact.pipeline, f"classification_model_{model_name}")
            elif hasattr(model_artifact, 'model'):
                # PyTorch/Transformers model
                mlflow.pytorch.log_model(model_artifact.model, f"classification_model_{model_name}")
        
        # Save additional artifacts
        if artifacts_dir:
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="classification_artifacts")
        
        logger.info(f"Classification experiment logged: {model_name}")
    
    def log_summarization_experiment(self,
                                   model_name: str,
                                   summarization_method: str,
                                   model_params: Dict[str, Any],
                                   evaluation_metrics: Dict[str, float],
                                   sample_summaries: Optional[List[Dict[str, str]]] = None,
                                   artifacts_dir: Optional[Path] = None):
        """
        Log summarization experiment results
        
        Args:
            model_name: Name of the summarization model
            summarization_method: Type of summarization (extractive/abstractive)
            model_params: Model parameters
            evaluation_metrics: Evaluation metrics
            sample_summaries: Sample summaries for qualitative analysis
            artifacts_dir: Directory to save artifacts
        """
        
        # Log model parameters
        mlflow.log_param("model_type", "summarization")
        mlflow.log_param("summarization_model", model_name)
        mlflow.log_param("summarization_method", summarization_method)
        
        for param, value in model_params.items():
            mlflow.log_param(f"summarization_{param}", value)
        
        # Log evaluation metrics
        for metric, value in evaluation_metrics.items():
            mlflow.log_metric(f"summarization_{metric}", value)
        
        # Log sample summaries
        if sample_summaries:
            samples_data = []
            for i, sample in enumerate(sample_summaries[:10]):  # Log top 10 samples
                sample_data = {
                    'sample_id': i,
                    'original_text': sample.get('original', ''),
                    'summary': sample.get('summary', ''),
                    'method': sample.get('method', summarization_method),
                    'compression_ratio': sample.get('compression_ratio', 0)
                }
                samples_data.append(sample_data)
            
            # Save as JSON artifact
            temp_file = Path("temp_summaries.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(samples_data, f, ensure_ascii=False, indent=2)
            
            mlflow.log_artifact(str(temp_file), artifact_path="summarization_samples")
            temp_file.unlink()  # Clean up
        
        # Save additional artifacts
        if artifacts_dir:
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="summarization_artifacts")
        
        logger.info(f"Summarization experiment logged: {model_name} ({summarization_method})")
    
    def log_hybrid_pipeline_experiment(self,
                                     pipeline_config: Dict[str, Any],
                                     training_results: Dict[str, Any],
                                     evaluation_results: Dict[str, Any],
                                     performance_stats: Dict[str, Any]):
        """
        Log complete hybrid pipeline experiment
        
        Args:
            pipeline_config: Pipeline configuration
            training_results: Training results from hybrid pipeline
            evaluation_results: Evaluation results on test set
            performance_stats: Performance statistics
        """
        
        # Log pipeline configuration
        mlflow.log_param("pipeline_type", "hybrid")
        mlflow.log_param("classifier_model", pipeline_config.get('classifier_model'))
        mlflow.log_param("summarization_model", pipeline_config.get('summarization_model'))
        mlflow.log_param("summarization_threshold", pipeline_config.get('summarization_threshold'))
        
        # Log training results
        if 'classifier_metrics' in training_results:
            for metric, value in training_results['classifier_metrics'].items():
                mlflow.log_metric(f"training_{metric}", value)
        
        mlflow.log_metric("training_time", training_results.get('training_time', 0))
        mlflow.log_param("training_samples", training_results.get('training_samples', 0))
        
        # Log evaluation results
        if 'classification' in evaluation_results:
            class_results = evaluation_results['classification']
            mlflow.log_metric("test_accuracy", class_results.get('accuracy', 0))
            
            # Log per-class metrics
            if 'classification_report' in class_results:
                class_report = class_results['classification_report']
                
                # Log macro and weighted averages
                for avg_type in ['macro avg', 'weighted avg']:
                    if avg_type in class_report:
                        metrics = class_report[avg_type]
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"test_{avg_type.replace(' ', '_')}_{metric}", value)
        
        # Log summarization evaluation
        if 'summarization' in evaluation_results:
            summ_results = evaluation_results['summarization']
            
            if summ_results.get('evaluated', False):
                mlflow.log_metric("avg_summarization_time", summ_results.get('avg_summarization_time', 0))
                mlflow.log_metric("avg_compression_ratio", summ_results.get('avg_compression_ratio', 0))
                mlflow.log_metric("summarization_sample_size", summ_results.get('sample_size', 0))
        
        # Log performance statistics
        for stat, value in performance_stats.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"performance_{stat}", value)
        
        logger.info("Hybrid pipeline experiment logged")
    
    def create_classification_report_artifact(self, y_true: np.ndarray, y_pred: np.ndarray,
                                            category_names: List[str],
                                            save_path: Optional[Path] = None) -> Path:
        """
        Create and save classification report visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            category_names: Names of categories
            save_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        
        if save_path is None:
            save_path = Path("classification_report.png")
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Confusion matrix heatmap
        plt.subplot(2, 1, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=category_names, yticklabels=category_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=category_names, output_dict=True)
        
        # Extract metrics for plotting
        categories = category_names
        precision = [class_report[cat]['precision'] for cat in categories]
        recall = [class_report[cat]['recall'] for cat in categories]
        f1_score = [class_report[cat]['f1-score'] for cat in categories]
        
        plt.subplot(2, 1, 2)
        x_pos = np.arange(len(categories))
        width = 0.25
        
        plt.bar(x_pos - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x_pos, recall, width, label='Recall', alpha=0.8)
        plt.bar(x_pos + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Categories')
        plt.ylabel('Score')
        plt.title('Classification Metrics by Category')
        plt.xticks(x_pos, categories, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_text_length_distribution_plot(self, text_lengths: List[int],
                                           save_path: Optional[Path] = None) -> Path:
        """
        Create text length distribution plot
        
        Args:
            text_lengths: List of text lengths
            save_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        
        if save_path is None:
            save_path = Path("text_length_distribution.png")
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Text Length Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(text_lengths, vert=True)
        plt.ylabel('Text Length (characters)')
        plt.title('Text Length Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def compare_models(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """
        Compare models across runs in the experiment
        
        Args:
            experiment_name: Name of experiment (uses current if None)
            
        Returns:
            DataFrame with model comparison
        """
        
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        # Get experiment
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return pd.DataFrame()
        
        # Get all runs
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        # Extract run data
        comparison_data = []
        
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unnamed'),
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'status': run.info.status
            }
            
            # Add parameters
            for param, value in run.data.params.items():
                run_data[f'param_{param}'] = value
            
            # Add metrics
            for metric, value in run.data.metrics.items():
                run_data[f'metric_{metric}'] = value
            
            comparison_data.append(run_data)
        
        df = pd.DataFrame(comparison_data)
        
        logger.info(f"Retrieved {len(df)} runs for comparison")
        
        return df
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        logger.info("MLflow run ended")


def main():
    """Example usage of UkrainianNLPExperimentTracker"""
    
    # Initialize tracker
    tracker = UkrainianNLPExperimentTracker(
        experiment_name="ukrainian_news_demo"
    )
    
    # Start a run
    run_id = tracker.start_run(
        run_name="demo_experiment",
        description="Demo of Ukrainian NLP pipeline tracking",
        tags={'demo': 'true', 'model': 'bert'}
    )
    
    # Log some example data
    data_info = {
        'dataset_name': 'FIdo-AI/ua-news',
        'train_samples': 120000,
        'test_samples': 30000,
        'categories': ['політика', 'спорт', 'новини', 'бізнес', 'технології'],
        'text_length_stats': {
            'mean': 850.0,
            'median': 650.0,
            'max': 5000,
            'std': 420.0
        }
    }
    
    tracker.log_data_info(data_info)
    
    # Log classification experiment
    model_params = {'model_type': 'bert', 'max_length': 512, 'num_epochs': 3}
    training_metrics = {'accuracy': 0.89, 'loss': 0.25}
    validation_metrics = {'accuracy': 0.87, 'loss': 0.28}
    
    tracker.log_classification_experiment(
        model_name="bert-multilingual",
        model_params=model_params,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics
    )
    
    # End run
    tracker.end_run()
    
    print(f"Demo experiment completed. Run ID: {run_id}")


if __name__ == "__main__":
    main()