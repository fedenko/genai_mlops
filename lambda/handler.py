import json
import boto3
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import tempfile
import zipfile

def lambda_handler(event, context):
    """
    AWS Lambda function for DistilBERT inference on AG News classification
    
    Expected event format:
    {
        "text": "News article text to classify"
    }
    
    Returns:
    {
        "statusCode": 200,
        "body": {
            "predicted_class": "World|Sports|Business|Sci/Tech",
            "confidence": 0.95,
            "probabilities": {
                "World": 0.95,
                "Sports": 0.02,
                "Business": 0.02,
                "Sci/Tech": 0.01
            }
        }
    }
    """
    
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
            
        text = body.get('text')
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required field: text'
                })
            }
        
        # Load model from S3 (this would typically be cached in practice)
        s3_bucket = os.environ.get('S3_BUCKET')
        model_path = download_model_from_s3(s3_bucket)
        
        # Load tokenizer and model
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # Tokenize input
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        # Map class IDs to labels
        class_labels = {
            0: "World",
            1: "Sports", 
            2: "Business",
            3: "Sci/Tech"
        }
        
        predicted_class = class_labels[predicted_class_id]
        
        # Format probabilities
        prob_dict = {
            class_labels[i]: probabilities[0][i].item() 
            for i in range(len(class_labels))
        }
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }

def download_model_from_s3(s3_bucket):
    """Download model artifacts from S3 to temporary directory"""
    s3_client = boto3.client('s3')
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # List all model files
    response = s3_client.list_objects_v2(
        Bucket=s3_bucket,
        Prefix='models/distilbert-agnews/'
    )
    
    if 'Contents' not in response:
        raise Exception("Model not found in S3")
    
    # Download all model files
    for obj in response['Contents']:
        key = obj['Key']
        local_path = os.path.join(temp_dir, os.path.basename(key))
        s3_client.download_file(s3_bucket, key, local_path)
    
    return temp_dir

# For local testing
if __name__ == "__main__":
    test_event = {
        "text": "Apple reported strong quarterly earnings with iPhone sales up 15%"
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))