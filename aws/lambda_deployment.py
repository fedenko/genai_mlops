"""
AWS Lambda Deployment Script for Ukrainian News Pipeline
Handles packaging, deployment, and configuration of Lambda function
"""

import boto3
import json
import zipfile
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class LambdaDeployer:
    """Handles deployment of Ukrainian NLP pipeline to AWS Lambda"""
    
    def __init__(self, aws_region: str = 'us-west-2'):
        """
        Initialize Lambda deployer
        
        Args:
            aws_region: AWS region for deployment
        """
        self.aws_region = aws_region
        
        # Initialize AWS clients
        self.lambda_client = boto3.client('lambda', region_name=aws_region)
        self.iam_client = boto3.client('iam', region_name=aws_region)
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.apigateway_client = boto3.client('apigateway', region_name=aws_region)
        
        # Configuration
        self.function_name = 'ukrainian-nlp-pipeline'
        self.handler = 'lambda_function.lambda_handler'
        self.runtime = 'python3.9'
        self.timeout = 300  # 5 minutes
        self.memory_size = 3008  # Maximum for better performance
        
    def create_iam_role(self) -> str:
        """Create IAM role for Lambda function"""
        
        role_name = f"{self.function_name}-role"
        
        # Trust policy for Lambda
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Check if role exists
            response = self.iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            logger.info(f"Using existing IAM role: {role_arn}")
            
        except self.iam_client.exceptions.NoSuchEntityException:
            # Create new role
            logger.info(f"Creating IAM role: {role_name}")
            
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='IAM role for Ukrainian NLP Lambda function'
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach basic Lambda execution policy
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            # Attach S3 access policy for model artifacts
            s3_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            "arn:aws:s3:::ukrainian-nlp-models/*",
                            "arn:aws:s3:::ukrainian-nlp-models"
                        ]
                    }
                ]
            }
            
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName='S3ModelAccess',
                PolicyDocument=json.dumps(s3_policy)
            )
            
            logger.info(f"Created IAM role: {role_arn}")
            
            # Wait for role to be ready
            time.sleep(10)
        
        return role_arn
    
    def create_deployment_package(self, source_dir: Path) -> Path:
        """
        Create deployment package for Lambda
        
        Args:
            source_dir: Source directory containing the code
            
        Returns:
            Path to the deployment package
        """
        
        logger.info("Creating Lambda deployment package...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            package_dir = temp_path / 'package'
            package_dir.mkdir()
            
            # Copy source code
            src_dir = source_dir / 'src'
            if src_dir.exists():
                shutil.copytree(src_dir, package_dir / 'src')
            
            # Copy Lambda function
            lambda_file = source_dir / 'src' / 'deployment' / 'lambda_function.py'
            if lambda_file.exists():
                shutil.copy2(lambda_file, package_dir / 'lambda_function.py')
            else:
                raise FileNotFoundError(f"Lambda function not found: {lambda_file}")
            
            # Install dependencies in package directory
            requirements_file = source_dir / 'requirements.txt'
            if requirements_file.exists():
                # Create minimal requirements for Lambda
                lambda_requirements = [
                    'torch>=2.0.0',
                    'transformers>=4.30.0',
                    'boto3>=1.28.0',
                    'numpy>=1.24.0'
                ]
                
                lambda_req_file = package_dir / 'requirements.txt'
                with open(lambda_req_file, 'w') as f:
                    f.write('\\n'.join(lambda_requirements))
                
                # Install packages
                logger.info("Installing Python dependencies...")
                subprocess.run([
                    'pip', 'install', '-r', str(lambda_req_file), 
                    '-t', str(package_dir),
                    '--no-deps',  # Skip dependencies to reduce size
                    '--platform', 'linux_x86_64',
                    '--implementation', 'cp',
                    '--only-binary=:all:'
                ], check=True)
            
            # Create zip file
            zip_path = temp_path / 'deployment-package.zip'
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)
            
            # Copy to permanent location
            final_zip_path = source_dir / 'deployment-package.zip'
            shutil.copy2(zip_path, final_zip_path)
            
            # Check package size
            package_size = final_zip_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Deployment package created: {package_size:.1f} MB")
            
            if package_size > 250:
                logger.warning("Package size exceeds 250MB. Consider reducing dependencies.")
            
            return final_zip_path
    
    def deploy_lambda_function(self, package_path: Path, role_arn: str,
                             environment_variables: Optional[Dict[str, str]] = None) -> str:
        """
        Deploy Lambda function
        
        Args:
            package_path: Path to deployment package
            role_arn: IAM role ARN
            environment_variables: Environment variables for function
            
        Returns:
            Function ARN
        """
        
        logger.info(f"Deploying Lambda function: {self.function_name}")
        
        # Default environment variables
        env_vars = {
            'MODEL_BUCKET': 'ukrainian-nlp-models',
            'MODEL_VERSION': 'latest',
            'CLASSIFIER_MODEL': 'bert',
            'SUMMARIZATION_THRESHOLD': '500'
        }
        
        if environment_variables:
            env_vars.update(environment_variables)
        
        # Read deployment package
        with open(package_path, 'rb') as f:
            zip_content = f.read()
        
        try:
            # Check if function exists
            response = self.lambda_client.get_function(FunctionName=self.function_name)
            
            # Update existing function
            logger.info("Updating existing Lambda function...")
            
            # Update function code
            self.lambda_client.update_function_code(
                FunctionName=self.function_name,
                ZipFile=zip_content
            )
            
            # Update function configuration
            response = self.lambda_client.update_function_configuration(
                FunctionName=self.function_name,
                Runtime=self.runtime,
                Role=role_arn,
                Handler=self.handler,
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Environment={'Variables': env_vars}
            )
            
        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            logger.info("Creating new Lambda function...")
            
            response = self.lambda_client.create_function(
                FunctionName=self.function_name,
                Runtime=self.runtime,
                Role=role_arn,
                Handler=self.handler,
                Code={'ZipFile': zip_content},
                Description='Ukrainian News Classification and Summarization Pipeline',
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Environment={'Variables': env_vars},
                Tags={
                    'Project': 'UkrainianNLP',
                    'Environment': 'Production',
                    'ManagedBy': 'MLOps'
                }
            )
        
        function_arn = response['FunctionArn']
        logger.info(f"Lambda function deployed: {function_arn}")
        
        return function_arn
    
    def create_api_gateway(self, function_arn: str) -> Dict[str, str]:
        """
        Create API Gateway for Lambda function
        
        Args:
            function_arn: Lambda function ARN
            
        Returns:
            Dictionary with API information
        """
        
        logger.info("Creating API Gateway...")
        
        api_name = f"{self.function_name}-api"
        
        # Create REST API
        try:
            # Check if API exists
            apis = self.apigateway_client.get_rest_apis()
            existing_api = None
            
            for api in apis['items']:
                if api['name'] == api_name:
                    existing_api = api
                    break
            
            if existing_api:
                api_id = existing_api['id']
                logger.info(f"Using existing API Gateway: {api_id}")
            else:
                # Create new API
                response = self.apigateway_client.create_rest_api(
                    name=api_name,
                    description='API for Ukrainian News NLP Pipeline',
                    endpointConfiguration={'types': ['REGIONAL']}
                )
                
                api_id = response['id']
                logger.info(f"Created API Gateway: {api_id}")
            
            # Get root resource
            resources = self.apigateway_client.get_resources(restApiId=api_id)
            root_resource_id = None
            
            for resource in resources['items']:
                if resource['path'] == '/':
                    root_resource_id = resource['id']
                    break
            
            # Create resource for processing
            try:
                process_resource = self.apigateway_client.create_resource(
                    restApiId=api_id,
                    parentId=root_resource_id,
                    pathPart='process'
                )
                process_resource_id = process_resource['id']
            except self.apigateway_client.exceptions.ConflictException:
                # Resource already exists
                for resource in resources['items']:
                    if resource['path'] == '/process':
                        process_resource_id = resource['id']
                        break
            
            # Create POST method
            try:
                self.apigateway_client.put_method(
                    restApiId=api_id,
                    resourceId=process_resource_id,
                    httpMethod='POST',
                    authorizationType='NONE'
                )
            except self.apigateway_client.exceptions.ConflictException:
                pass  # Method already exists
            
            # Set up integration
            lambda_uri = f"arn:aws:apigateway:{self.aws_region}:lambda:path/2015-03-31/functions/{function_arn}/invocations"
            
            try:
                self.apigateway_client.put_integration(
                    restApiId=api_id,
                    resourceId=process_resource_id,
                    httpMethod='POST',
                    type='AWS_PROXY',
                    integrationHttpMethod='POST',
                    uri=lambda_uri
                )
            except self.apigateway_client.exceptions.ConflictException:
                pass  # Integration already exists
            
            # Add Lambda permission for API Gateway
            try:
                self.lambda_client.add_permission(
                    FunctionName=self.function_name,
                    StatementId='api-gateway-invoke',
                    Action='lambda:InvokeFunction',
                    Principal='apigateway.amazonaws.com',
                    SourceArn=f"arn:aws:execute-api:{self.aws_region}:*:{api_id}/*/*/*"
                )
            except self.lambda_client.exceptions.ResourceConflictException:
                pass  # Permission already exists
            
            # Deploy API
            deployment = self.apigateway_client.create_deployment(
                restApiId=api_id,
                stageName='prod',
                description='Production deployment'
            )
            
            # Construct API URL
            api_url = f"https://{api_id}.execute-api.{self.aws_region}.amazonaws.com/prod/process"
            
            logger.info(f"API Gateway deployed: {api_url}")
            
            return {
                'api_id': api_id,
                'api_url': api_url,
                'stage': 'prod'
            }
            
        except Exception as e:
            logger.error(f"Failed to create API Gateway: {e}")
            raise
    
    def deploy_complete_pipeline(self, source_dir: str,
                               environment_variables: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Deploy complete pipeline (Lambda + API Gateway)
        
        Args:
            source_dir: Source directory path
            environment_variables: Environment variables for Lambda
            
        Returns:
            Deployment information
        """
        
        source_path = Path(source_dir)
        
        logger.info("Starting complete pipeline deployment...")
        
        try:
            # 1. Create IAM role
            role_arn = self.create_iam_role()
            
            # 2. Create deployment package
            package_path = self.create_deployment_package(source_path)
            
            # 3. Deploy Lambda function
            function_arn = self.deploy_lambda_function(
                package_path, role_arn, environment_variables
            )
            
            # 4. Create API Gateway
            api_info = self.create_api_gateway(function_arn)
            
            # 5. Test deployment
            logger.info("Testing deployment...")
            
            # Wait for deployment to be ready
            time.sleep(10)
            
            test_result = self.test_deployment(api_info['api_url'])
            
            deployment_info = {
                'function_name': self.function_name,
                'function_arn': function_arn,
                'api_url': api_info['api_url'],
                'api_id': api_info['api_id'],
                'region': self.aws_region,
                'test_result': test_result
            }
            
            logger.info("Deployment completed successfully!")
            logger.info(f"API URL: {api_info['api_url']}")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    def test_deployment(self, api_url: str) -> Dict[str, any]:
        """Test the deployed API"""
        
        import requests
        
        test_payload = {
            'title': 'Тестова новина',
            'text': 'Це тестовий текст для перевірки роботи системи класифікації та реферування українських новин. Система повинна правильно визначити категорію та створити стислий опис.',
            'include_summarization': True
        }
        
        try:
            response = requests.post(
                api_url,
                json=test_payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'status': 'success',
                    'response_time': response.elapsed.total_seconds(),
                    'classification': result.get('classification', {}),
                    'summarization': bool(result.get('summarization'))
                }
            else:
                return {
                    'status': 'error',
                    'status_code': response.status_code,
                    'error': response.text
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


def main():
    """Main deployment function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Ukrainian NLP Pipeline to AWS Lambda')
    parser.add_argument('--source-dir', default='.', help='Source directory path')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--model-bucket', help='S3 bucket for model artifacts')
    parser.add_argument('--classifier-model', default='bert', choices=['bert', 'svm'], 
                       help='Classifier model type')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Environment variables
    env_vars = {}
    if args.model_bucket:
        env_vars['MODEL_BUCKET'] = args.model_bucket
    if args.classifier_model:
        env_vars['CLASSIFIER_MODEL'] = args.classifier_model
    
    # Deploy pipeline
    deployer = LambdaDeployer(aws_region=args.region)
    
    deployment_info = deployer.deploy_complete_pipeline(
        source_dir=args.source_dir,
        environment_variables=env_vars
    )
    
    # Print deployment information
    print("\\n" + "="*60)
    print("DEPLOYMENT COMPLETE")
    print("="*60)
    print(f"Function Name: {deployment_info['function_name']}")
    print(f"Function ARN: {deployment_info['function_arn']}")
    print(f"API URL: {deployment_info['api_url']}")
    print(f"Region: {deployment_info['region']}")
    print(f"Test Result: {deployment_info['test_result']['status']}")
    print("="*60)
    
    # Save deployment info
    with open('deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("Deployment information saved to deployment_info.json")


if __name__ == "__main__":
    main()