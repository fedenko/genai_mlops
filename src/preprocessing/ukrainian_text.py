"""
Ukrainian Text Preprocessing Module
Handles Cyrillic text cleaning, tokenization, and feature extraction
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UkrainianTextProcessor:
    """Specialized text processor for Ukrainian language"""
    
    def __init__(self):
        # Ukrainian stopwords (common words to remove)
        self.ukrainian_stopwords = {
            'а', 'але', 'ба', 'бо', 'в', 'ви', 'до', 'за', 'з', 'і', 'із', 'к', 'ко', 
            'на', 'не', 'ні', 'о', 'об', 'од', 'по', 'та', 'то', 'у', 'як', 'що', 'це',
            'або', 'адже', 'би', 'була', 'було', 'були', 'буде', 'будуть', 'вас', 'ваш',
            'все', 'вже', 'для', 'його', 'його', 'її', 'його', 'коли', 'може', 'нас',
            'наш', 'них', 'про', 'свій', 'так', 'там', 'тут', 'хто', 'чи', 'цей', 'цього'
        }
        
        # Cyrillic character ranges
        self.cyrillic_pattern = re.compile(r'[а-яё]', re.IGNORECASE)
        
        # Common Ukrainian patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'\+?3?8?0\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}')
        
    def clean_text(self, text: str) -> str:
        """
        Clean Ukrainian text by removing unwanted characters and normalizing
        
        Args:
            text: Raw Ukrainian text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Remove English punctuation but keep Ukrainian-specific characters
        # Keep: apostrophe ('), dash (-), and Ukrainian quotation marks
        ukrainian_punct = string.punctuation.replace("'", "").replace("-", "")
        text = text.translate(str.maketrans('', '', ukrainian_punct))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize Ukrainian text into words
        
        Args:
            text: Cleaned Ukrainian text
            remove_stopwords: Whether to remove Ukrainian stopwords
            
        Returns:
            List of tokens
        """
        # Basic word tokenization
        tokens = text.split()
        
        # Filter out very short tokens and non-Cyrillic words
        tokens = [
            token for token in tokens 
            if len(token) > 2 and self.cyrillic_pattern.search(token)
        ]
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.ukrainian_stopwords]
        
        return tokens
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract text features specific to Ukrainian language
        
        Args:
            text: Ukrainian text
            
        Returns:
            Dictionary of features
        """
        clean_text = self.clean_text(text)
        tokens = self.tokenize(clean_text, remove_stopwords=False)
        
        features = {
            'text_length': len(text),
            'clean_text_length': len(clean_text),
            'word_count': len(tokens),
            'avg_word_length': np.mean([len(token) for token in tokens]) if tokens else 0,
            'cyrillic_ratio': len(self.cyrillic_pattern.findall(text)) / len(text) if text else 0,
            'stopword_ratio': len([t for t in tokens if t in self.ukrainian_stopwords]) / len(tokens) if tokens else 0,
            'unique_word_ratio': len(set(tokens)) / len(tokens) if tokens else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        return features
    
    def preprocess_dataset(self, df: pd.DataFrame, 
                          text_columns: List[str] = ['title', 'text']) -> pd.DataFrame:
        """
        Preprocess entire dataset of Ukrainian news
        
        Args:
            df: DataFrame with Ukrainian text
            text_columns: Columns containing text to preprocess
            
        Returns:
            DataFrame with additional preprocessed columns
        """
        df = df.copy()
        
        logger.info(f"Preprocessing {len(df)} samples with columns: {text_columns}")
        
        for col in text_columns:
            if col in df.columns:
                # Create cleaned version
                df[f'{col}_clean'] = df[col].apply(self.clean_text)
                
                # Create tokenized version (as string for storage)
                df[f'{col}_tokens'] = df[col].apply(
                    lambda x: ' '.join(self.tokenize(self.clean_text(x)))
                )
                
                # Extract features
                features_list = df[col].apply(self.extract_features)
                feature_df = pd.DataFrame(features_list.tolist())
                
                # Add features with column prefix
                for feature_col in feature_df.columns:
                    df[f'{col}_{feature_col}'] = feature_df[feature_col]
        
        # Combined text processing (title + text)
        if 'title' in df.columns and 'text' in df.columns:
            df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            df['combined_clean'] = df['combined_text'].apply(self.clean_text)
            df['combined_tokens'] = df['combined_text'].apply(
                lambda x: ' '.join(self.tokenize(self.clean_text(x)))
            )
        
        logger.info(f"Preprocessing complete. Added {len([c for c in df.columns if '_clean' in c or '_tokens' in c or 'combined' in c])} new columns")
        
        return df
    
    def get_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from Ukrainian texts
        
        Args:
            texts: List of Ukrainian texts
            min_freq: Minimum frequency for word to be included
            
        Returns:
            Dictionary mapping words to frequencies
        """
        word_counts = {}
        
        for text in texts:
            tokens = self.tokenize(self.clean_text(text))
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Filter by minimum frequency
        vocabulary = {
            word: count for word, count in word_counts.items() 
            if count >= min_freq
        }
        
        logger.info(f"Built vocabulary of {len(vocabulary)} words (min_freq={min_freq})")
        
        return vocabulary


def main():
    """Example usage of UkrainianTextProcessor"""
    
    processor = UkrainianTextProcessor()
    
    # Example Ukrainian text
    sample_text = """
    Це приклад українського тексту для новин. 
    Сьогодні в Києві відбулася важлива подія!
    Чи знаєте ви про це?
    """
    
    print("Original text:", sample_text)
    print("Cleaned text:", processor.clean_text(sample_text))
    print("Tokens:", processor.tokenize(processor.clean_text(sample_text)))
    print("Features:", processor.extract_features(sample_text))


if __name__ == "__main__":
    main()