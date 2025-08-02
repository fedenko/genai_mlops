"""
Ukrainian Text Summarization Models
Supports extractive and abstractive summarization for Ukrainian news
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle
import json
import re

# NLP libraries
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Deep Learning
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline as hf_pipeline
)

# Evaluation
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class ExtractiveSummarizer:
    """Extractive summarization using TF-IDF and sentence ranking"""
    
    def __init__(self, language: str = 'ukrainian'):
        self.language = language
        
        # Ukrainian sentence patterns
        self.sentence_end_pattern = re.compile(r'[.!?]+')
        
        # TF-IDF vectorizer for sentence ranking
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words=None  # We handle Ukrainian stopwords in preprocessing
        )
        
        # Ukrainian stopwords for sentence filtering
        self.ukrainian_stopwords = {
            'а', 'але', 'ба', 'бо', 'в', 'ви', 'до', 'за', 'з', 'і', 'із', 'к', 'ко', 
            'на', 'не', 'ні', 'о', 'об', 'од', 'по', 'та', 'то', 'у', 'як', 'що', 'це',
            'або', 'адже', 'би', 'була', 'було', 'були', 'буде', 'будуть', 'вас', 'ваш',
            'все', 'вже', 'для', 'його', 'його', 'її', 'його', 'коли', 'може', 'нас',
            'наш', 'них', 'про', 'свій', 'так', 'там', 'тут', 'хто', 'чи', 'цей', 'цього'
        }
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split Ukrainian text into sentences"""
        
        # Basic sentence splitting by punctuation
        sentences = self.sentence_end_pattern.split(text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter out very short sentences or non-Ukrainian content
            if len(sentence) > 20 and any(ord(char) >= 1040 and ord(char) <= 1103 for char in sentence):
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def calculate_sentence_scores(self, sentences: List[str]) -> np.ndarray:
        """Calculate TF-IDF based scores for sentences"""
        
        if len(sentences) < 2:
            return np.array([1.0] * len(sentences))
        
        try:
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores as sum of TF-IDF values
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Normalize scores
            if sentence_scores.max() > 0:
                sentence_scores = sentence_scores / sentence_scores.max()
            
            return sentence_scores
            
        except Exception as e:
            logger.warning(f"Failed to calculate TF-IDF scores: {e}")
            return np.array([1.0] * len(sentences))
    
    def rank_sentences(self, sentences: List[str], scores: np.ndarray, 
                      num_sentences: int = 3) -> List[Tuple[int, str, float]]:
        """Rank sentences by scores and return top sentences"""
        
        # Create sentence ranking
        sentence_ranking = []
        for i, (sentence, score) in enumerate(zip(sentences, scores)):
            sentence_ranking.append((i, sentence, score))
        
        # Sort by score (descending)
        sentence_ranking.sort(key=lambda x: x[2], reverse=True)
        
        # Return top sentences
        return sentence_ranking[:num_sentences]
    
    def summarize(self, text: str, max_sentences: int = 3, 
                 min_length: int = 50) -> Dict[str, Any]:
        """
        Generate extractive summary of Ukrainian text
        
        Args:
            text: Input Ukrainian text
            max_sentences: Maximum number of sentences in summary
            min_length: Minimum length of input text to summarize
            
        Returns:
            Dictionary with summary and metadata
        """
        
        if len(text) < min_length:
            return {
                'summary': text,
                'method': 'extractive',
                'num_sentences': 1,
                'compression_ratio': 1.0,
                'original_length': len(text),
                'summary_length': len(text)
            }
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= max_sentences:
            summary = '. '.join(sentences) + '.'
            return {
                'summary': summary,
                'method': 'extractive',
                'num_sentences': len(sentences),
                'compression_ratio': 1.0,
                'original_length': len(text),
                'summary_length': len(summary)
            }
        
        # Calculate sentence scores
        scores = self.calculate_sentence_scores(sentences)
        
        # Rank and select top sentences
        top_sentences = self.rank_sentences(sentences, scores, max_sentences)
        
        # Sort selected sentences by original order
        selected_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        # Create summary
        summary_sentences = [sent[1] for sent in selected_sentences]
        summary = '. '.join(summary_sentences) + '.'
        
        return {
            'summary': summary,
            'method': 'extractive',
            'num_sentences': len(summary_sentences),
            'compression_ratio': len(summary) / len(text),
            'original_length': len(text),
            'summary_length': len(summary),
            'sentence_scores': [sent[2] for sent in selected_sentences]
        }


class AbstractiveSummarizer:
    """Abstractive summarization using T5/mT5 models"""
    
    def __init__(self, model_name: str = "ukr-models/uk-summarizer"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        self.load_model()
    
    def load_model(self):
        """Load the summarization model"""
        
        try:
            logger.info(f"Loading summarization model: {self.model_name}")
            
            # Try loading Ukrainian-specific model first
            if "ukr-models" in self.model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            else:
                # Fallback to multilingual T5
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline
            self.pipeline = hf_pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Ukrainian model, falling back to mT5: {e}")
            
            # Fallback to multilingual T5
            fallback_model = "google/mt5-small"
            self.model_name = fallback_model
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
            
            self.pipeline = hf_pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> Dict[str, Any]:
        """
        Generate abstractive summary using T5/mT5
        
        Args:
            text: Input Ukrainian text
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Dictionary with summary and metadata
        """
        
        if len(text) < min_length:
            return {
                'summary': text,
                'method': 'abstractive',
                'model': self.model_name,
                'compression_ratio': 1.0,
                'original_length': len(text),
                'summary_length': len(text)
            }
        
        try:
            # Prepare input for T5 (add task prefix if needed)
            input_text = text
            if "t5" in self.model_name.lower():
                input_text = f"summarize: {text}"
            
            # Generate summary
            summary_result = self.pipeline(
                input_text,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                clean_up_tokenization_spaces=True
            )
            
            summary = summary_result[0]['summary_text']
            
            return {
                'summary': summary,
                'method': 'abstractive',
                'model': self.model_name,
                'compression_ratio': len(summary) / len(text),
                'original_length': len(text),
                'summary_length': len(summary)
            }
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            
            # Fallback to simple truncation
            words = text.split()
            if len(words) > 50:
                summary = ' '.join(words[:50]) + '...'
            else:
                summary = text
            
            return {
                'summary': summary,
                'method': 'fallback_truncation',
                'model': 'none',
                'compression_ratio': len(summary) / len(text),
                'original_length': len(text),
                'summary_length': len(summary),
                'error': str(e)
            }


class HybridSummarizer:
    """Hybrid summarizer combining extractive and abstractive approaches"""
    
    def __init__(self, ukrainian_model: str = "ukr-models/uk-summarizer"):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer(ukrainian_model)
        
        # ROUGE scorer for evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def summarize(self, text: str, method: str = 'auto', 
                 max_length: int = 150) -> Dict[str, Any]:
        """
        Generate summary using specified method or automatic selection
        
        Args:
            text: Input Ukrainian text
            method: 'extractive', 'abstractive', or 'auto'
            max_length: Maximum summary length
            
        Returns:
            Dictionary with summary and metadata
        """
        
        if method == 'extractive':
            return self.extractive.summarize(text, max_sentences=3)
        
        elif method == 'abstractive':
            return self.abstractive.summarize(text, max_length=max_length)
        
        elif method == 'auto':
            # Automatic method selection based on text characteristics
            text_length = len(text)
            
            if text_length < 200:
                # Too short for summarization
                return {
                    'summary': text,
                    'method': 'no_summarization',
                    'reason': 'text_too_short',
                    'compression_ratio': 1.0,
                    'original_length': text_length,
                    'summary_length': text_length
                }
            
            elif text_length < 1000:
                # Use extractive for shorter texts
                result = self.extractive.summarize(text, max_sentences=2)
                result['auto_method_selected'] = 'extractive'
                return result
            
            else:
                # Use abstractive for longer texts
                result = self.abstractive.summarize(text, max_length=max_length)
                result['auto_method_selected'] = 'abstractive'
                return result
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_methods(self, text: str, reference_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare extractive and abstractive summarization methods
        
        Args:
            text: Input text
            reference_summary: Optional reference summary for ROUGE evaluation
            
        Returns:
            Dictionary comparing both methods
        """
        
        # Generate summaries with both methods
        extractive_result = self.extractive.summarize(text)
        abstractive_result = self.abstractive.summarize(text)
        
        comparison = {
            'extractive': extractive_result,
            'abstractive': abstractive_result,
            'text_length': len(text)
        }
        
        # Calculate ROUGE scores if reference is provided
        if reference_summary:
            extractive_rouge = self.rouge_scorer.score(reference_summary, extractive_result['summary'])
            abstractive_rouge = self.rouge_scorer.score(reference_summary, abstractive_result['summary'])
            
            comparison['rouge_scores'] = {
                'extractive': {
                    'rouge1': extractive_rouge['rouge1'].fmeasure,
                    'rouge2': extractive_rouge['rouge2'].fmeasure,
                    'rougeL': extractive_rouge['rougeL'].fmeasure
                },
                'abstractive': {
                    'rouge1': abstractive_rouge['rouge1'].fmeasure,
                    'rouge2': abstractive_rouge['rouge2'].fmeasure,
                    'rougeL': abstractive_rouge['rougeL'].fmeasure
                }
            }
        
        return comparison
    
    def evaluate_summary_quality(self, text: str, summary: str) -> Dict[str, float]:
        """
        Evaluate summary quality using various metrics
        
        Args:
            text: Original text
            summary: Generated summary
            
        Returns:
            Dictionary with quality metrics
        """
        
        metrics = {}
        
        # Basic metrics
        metrics['compression_ratio'] = len(summary) / len(text)
        metrics['length_ratio'] = len(summary.split()) / len(text.split())
        
        # Lexical overlap (simple measure)
        text_words = set(text.lower().split())
        summary_words = set(summary.lower().split())
        
        if text_words:
            metrics['lexical_overlap'] = len(text_words & summary_words) / len(text_words)
        else:
            metrics['lexical_overlap'] = 0.0
        
        # Coverage (how much of the original text is covered)
        if summary_words:
            metrics['coverage'] = len(text_words & summary_words) / len(summary_words)
        else:
            metrics['coverage'] = 0.0
        
        return metrics


def main():
    """Example usage of Ukrainian summarizers"""
    
    # Sample Ukrainian text
    sample_text = """
    Україна продовжує захищати свою територіальну цілісність та суверенітет. 
    Сьогодні президент зустрівся з міжнародними партнерами для обговорення 
    підтримки України. Розглядалися питання військової та гуманітарної допомоги. 
    Також обговорювалася можливість додаткових санкцій проти агресора. 
    Міжнародна спільнота продовжує підтримувати Україну в цей складний час. 
    Важливо зберегти єдність та солідарність у боротьбі за мир та справедливість.
    """
    
    # Initialize hybrid summarizer
    summarizer = HybridSummarizer()
    
    # Test different methods
    print("Original text length:", len(sample_text))
    print("\nExtractive summary:")
    extractive_result = summarizer.summarize(sample_text, method='extractive')
    print(extractive_result['summary'])
    print(f"Compression ratio: {extractive_result['compression_ratio']:.2f}")
    
    print("\nAbstractive summary:")
    abstractive_result = summarizer.summarize(sample_text, method='abstractive')
    print(abstractive_result['summary'])
    print(f"Compression ratio: {abstractive_result['compression_ratio']:.2f}")
    
    print("\nAuto method:")
    auto_result = summarizer.summarize(sample_text, method='auto')
    print(auto_result['summary'])
    print(f"Selected method: {auto_result.get('auto_method_selected', 'N/A')}")


if __name__ == "__main__":
    main()