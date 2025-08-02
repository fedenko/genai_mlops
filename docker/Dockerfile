# Dockerfile for Ukrainian News NLP Pipeline
# Based on Python 3.9 with Ukrainian language support

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANG=uk_UA.UTF-8
ENV LANGUAGE=uk_UA:uk
ENV LC_ALL=uk_UA.UTF-8

# Install system dependencies including Ukrainian language support
RUN apt-get update && apt-get install -y \
    language-pack-uk \
    locales \
    locales-all \
    build-essential \
    git \
    curl \
    wget \
    && locale-gen uk_UA.UTF-8 \
    && update-locale LANG=uk_UA.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Ukrainian language processing
RUN python -c "import nltk; nltk.download('punkt')" \
    && python -c "import nltk; nltk.download('stopwords')" \
    && python -c "import spacy; spacy.cli.download('xx_core_web_sm')"

# Download and cache Ukrainian/multilingual models
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased'); \
    AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-multilingual-cased')"

# Try to download Ukrainian-specific model (graceful fallback if not available)
RUN python -c "try: \
    from transformers import AutoTokenizer, T5ForConditionalGeneration; \
    AutoTokenizer.from_pretrained('ukr-models/uk-summarizer'); \
    T5ForConditionalGeneration.from_pretrained('ukr-models/uk-summarizer'); \
    print('Ukrainian model downloaded successfully') \
    except Exception as e: \
    print(f'Ukrainian model not available, will use multilingual fallback: {e}')" || true

# Copy application code
COPY src/ /app/src/
COPY data/ /app/data/
COPY notebooks/ /app/notebooks/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs /app/mlflow

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlops
RUN chown -R mlops:mlops /app
USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "src/training/train_pipeline.py"]

# Alternative commands for different use cases:
# For interactive training: docker run -it <image> bash
# For Jupyter notebook: docker run -p 8888:8888 <image> jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# For MLflow server: docker run -p 5000:5000 <image> mlflow server --host 0.0.0.0 --port 5000