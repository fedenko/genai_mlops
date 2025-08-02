"""
AWS Lambda Function for Ukrainian News Pipeline Inference
Handles real-time classification and summarization requests
"""

import json
import logging
import time
from typing import Dict, Any, Optional
import boto3
from pathlib import Path
import pickle
import os

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for model caching
_classifier = None
_summarizer = None
_hybrid_pipeline = None
_model_artifacts = {}

# S3 client for model loading
s3_client = boto3.client('s3')

# Configuration from environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'ukrainian-nlp-models')
MODEL_VERSION = os.environ.get('MODEL_VERSION', 'latest')
CLASSIFIER_MODEL = os.environ.get('CLASSIFIER_MODEL', 'bert')
SUMMARIZATION_THRESHOLD = int(os.environ.get('SUMMARIZATION_THRESHOLD', '500'))


class ModelLoader:
    """Handles loading and caching of ML models in Lambda"""
    
    @staticmethod
    def download_from_s3(bucket: str, key: str, local_path: str) -> bool:
        """Download model artifact from S3"""
        try:
            s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded {key} from S3 to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {key} from S3: {e}")
            return False
    
    @staticmethod
    def load_classifier(model_type: str = 'bert') -> Any:
        """Load classification model"""
        global _classifier
        
        if _classifier is not None:
            return _classifier
        
        try:
            if model_type == 'bert':
                # Load BERT model from S3
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                
                model_dir = '/tmp/bert_classifier'
                os.makedirs(model_dir, exist_ok=True)
                
                # Download model files
                model_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'vocab.txt']
                
                for file_name in model_files:
                    s3_key = f"models/classifier_bert/{MODEL_VERSION}/{file_name}"
                    local_path = f"{model_dir}/{file_name}"
                    
                    if not ModelLoader.download_from_s3(MODEL_BUCKET, s3_key, local_path):
                        logger.warning(f"Could not download {file_name}, using default model")
                        # Fallback to default multilingual BERT
                        _classifier = pipeline(
                            "text-classification",
                            model="google-bert/bert-base-multilingual-cased",
                            device=-1  # CPU inference
                        )
                        return _classifier
                
                # Load downloaded model
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                
                _classifier = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1
                )
                
            else:
                # Load classical ML model
                model_path = '/tmp/classical_classifier.pkl'
                s3_key = f"models/classifier_{model_type}/{MODEL_VERSION}/model.pkl"
                
                if ModelLoader.download_from_s3(MODEL_BUCKET, s3_key, model_path):
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    _classifier = model_data['pipeline']
                else:
                    raise Exception(f"Could not load {model_type} classifier")
            
            logger.info(f"Classifier ({model_type}) loaded successfully")
            return _classifier
            
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise
    
    @staticmethod
    def load_summarizer() -> Any:
        """Load summarization model"""
        global _summarizer
        
        if _summarizer is not None:
            return _summarizer
        
        try:
            from transformers import pipeline
            
            # Try to load Ukrainian model, fallback to multilingual
            try:
                _summarizer = pipeline(
                    "summarization",
                    model="ukr-models/uk-summarizer",
                    device=-1
                )
                logger.info("Ukrainian summarizer loaded successfully")
            except Exception as e:
                logger.warning(f"Ukrainian model not available, using mT5: {e}")
                _summarizer = pipeline(
                    "summarization",
                    model="google/mt5-small",
                    device=-1
                )
                logger.info("Multilingual T5 summarizer loaded as fallback")
            
            return _summarizer
            
        except Exception as e:
            logger.error(f"Failed to load summarizer: {e}")
            raise
    
    @staticmethod
    def load_text_processor():
        """Load Ukrainian text processor"""
        try:
            # Simple text processing for Lambda (reduced dependencies)
            import re
            
            class SimplifiedUkrainianProcessor:
                def __init__(self):
                    self.cyrillic_pattern = re.compile(r'[а-яё]', re.IGNORECASE)
                    self.ukrainian_stopwords = {
                        'а', 'але', 'ба', 'бо', 'в', 'ви', 'до', 'за', 'з', 'і', 'із', 'к', 'ко', 
                        'на', 'не', 'ні', 'о', 'об', 'од', 'по', 'та', 'то', 'у', 'як', 'що', 'це'
                    }
                
                def clean_text(self, text: str) -> str:
                    if not isinstance(text, str):
                        return ""
                    
                    text = text.lower()
                    text = re.sub(r'http[s]?://\S+', ' ', text)
                    text = re.sub(r'\S+@\S+', ' ', text)
                    text = re.sub(r'\s+', ' ', text)
                    return text.strip()
                
                def tokenize(self, text: str) -> list:
                    tokens = text.split()
                    tokens = [
                        token for token in tokens 
                        if len(token) > 2 and self.cyrillic_pattern.search(token)
                    ]
                    tokens = [token for token in tokens if token not in self.ukrainian_stopwords]
                    return tokens
            
            return SimplifiedUkrainianProcessor()
            
        except Exception as e:
            logger.error(f"Failed to load text processor: {e}")
            raise


def lambda_handler(event, context):
    """
    Main Lambda handler for Ukrainian news processing
    
    Expected event structure:
    {
        "title": "Ukrainian news title",
        "text": "Ukrainian news content",
        "include_summarization": true,
        "summarization_method": "auto"
    }
    """
    
    start_time = time.time()
    
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        title = body.get('title', '')
        text = body.get('text', '')
        include_summarization = body.get('include_summarization', True)
        summarization_method = body.get('summarization_method', 'auto')
        
        if not title and not text:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json; charset=utf-8',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Either title or text must be provided'
                }, ensure_ascii=False)
            }
        
        # Load models (cached after first call)
        classifier = ModelLoader.load_classifier(CLASSIFIER_MODEL)
        text_processor = ModelLoader.load_text_processor()
        
        # Preprocess text
        combined_text = f"{title} {text}"
        clean_text = text_processor.clean_text(combined_text)
        tokens = ' '.join(text_processor.tokenize(clean_text))
        
        # Classification
        classification_start = time.time()
        
        if CLASSIFIER_MODEL == 'bert':
            # BERT classification
            classification_result = classifier(tokens)
            
            if isinstance(classification_result, list):
                classification_result = classification_result[0]
            
            # Extract label and score
            predicted_label = classification_result.get('label', 'UNKNOWN')
            confidence = classification_result.get('score', 0.0)
            
            # Map label to category (assuming LABEL_0, LABEL_1, etc.)
            category_mapping = {
                'LABEL_0': 'політика',
                'LABEL_1': 'спорт', 
                'LABEL_2': 'новини',
                'LABEL_3': 'бізнес',
                'LABEL_4': 'технології'
            }
            
            category = category_mapping.get(predicted_label, predicted_label)
            
        else:
            # Classical ML classification
            prediction = classifier.predict([tokens])[0]
            probabilities = classifier.predict_proba([tokens])[0] if hasattr(classifier, 'predict_proba') else [1.0]
            
            category_mapping = ['політика', 'спорт', 'новини', 'бізнес', 'технології']
            category = category_mapping[prediction] if prediction < len(category_mapping) else str(prediction)
            confidence = float(max(probabilities))
        
        classification_time = time.time() - classification_start
        
        # Summarization (conditional)
        summary_result = None
        summarization_time = 0
        
        if include_summarization and len(text) >= SUMMARIZATION_THRESHOLD:
            summarization_start = time.time()
            
            try:
                summarizer = ModelLoader.load_summarizer()
                
                # Prepare input for summarization
                summary_input = text
                if "t5" in str(type(summarizer.model)).lower():
                    summary_input = f"summarize: {text}"
                
                # Generate summary
                summary_output = summarizer(
                    summary_input,
                    max_length=150,
                    min_length=50,
                    num_beams=2,  # Reduced for faster inference
                    no_repeat_ngram_size=2,
                    clean_up_tokenization_spaces=True
                )
                
                if isinstance(summary_output, list):
                    summary_output = summary_output[0]
                
                summary_text = summary_output.get('summary_text', summary_output.get('generated_text', ''))
                
                summary_result = {
                    'summary': summary_text,
                    'method': 'abstractive',
                    'compression_ratio': len(summary_text) / len(text),
                    'original_length': len(text),
                    'summary_length': len(summary_text)
                }
                
                summarization_time = time.time() - summarization_start
                
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
                
                # Fallback to simple truncation
                words = text.split()
                if len(words) > 50:
                    summary_text = ' '.join(words[:50]) + '...'
                else:
                    summary_text = text
                
                summary_result = {
                    'summary': summary_text,
                    'method': 'fallback_truncation',
                    'compression_ratio': len(summary_text) / len(text),
                    'original_length': len(text),
                    'summary_length': len(summary_text),
                    'error': str(e)
                }
                
                summarization_time = time.time() - summarization_start
        
        total_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'classification': {
                'category': category,
                'confidence': float(confidence),
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
                'model_version': MODEL_VERSION,
                'classifier_model': CLASSIFIER_MODEL
            }
        }
        
        logger.info(f"Processed request: {category} ({confidence:.3f}) in {total_time:.3f}s")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json; charset=utf-8',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data, ensure_ascii=False)
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json; charset=utf-8',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error during processing'
            }, ensure_ascii=False)
        }


def health_check_handler(event, context):
    """Health check endpoint for Lambda"""
    
    try:
        # Basic health check
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json; charset=utf-8'
            },
            'body': json.dumps({
                'status': 'healthy',
                'model_version': MODEL_VERSION,
                'classifier_model': CLASSIFIER_MODEL,
                'summarization_threshold': SUMMARIZATION_THRESHOLD
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json; charset=utf-8'
            },
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e)
            })
        }


# For testing locally
if __name__ == "__main__":
    # Sample test event
    test_event = {
        'title': 'Українські новини сьогодні',
        'text': 'Це приклад українського тексту для тестування системи класифікації та реферування новин. ' * 10,
        'include_summarization': True,
        'summarization_method': 'auto'
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result['body']), indent=2, ensure_ascii=False))