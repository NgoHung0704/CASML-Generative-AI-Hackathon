"""
Data ingestion module - Loads and preprocesses corpus and Q&A datasets
"""

import os
import re
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import json


class CorpusLoader:
    """
    Loads and preprocesses text corpus (books, documents).
    """
    
    def __init__(self, config):
        """
        Initialize corpus loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.corpus_path = config.get('paths.corpus_file')
        self.encoding = config.get('ingestion.encoding', 'utf-8')
    
    def load_corpus(self) -> str:
        """
        Load corpus from file.
        
        Returns:
            Raw corpus text
        """
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        
        with open(self.corpus_path, 'r', encoding=self.encoding) as f:
            corpus = f.read()
        
        return corpus
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess raw text.
        
        Args:
            text: Raw text
        
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        if self.config.get('ingestion.remove_extra_whitespace', True):
            text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase
        if self.config.get('ingestion.lowercase', False):
            text = text.lower()
        
        # Remove special characters (optional)
        if self.config.get('ingestion.remove_special_chars', False):
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'"()]', '', text)
        
        return text.strip()
    
    def load_and_preprocess(self) -> str:
        """
        Load and preprocess corpus in one step.
        
        Returns:
            Preprocessed corpus text
        """
        corpus = self.load_corpus()
        return self.preprocess_text(corpus)


class QADataLoader:
    """
    Loads Q&A datasets (train/test).
    """
    
    def __init__(self, config):
        """
        Initialize Q&A data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.train_path = config.get('paths.train_questions')
        self.test_path = config.get('paths.test_questions')
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training Q&A data.
        
        Returns:
            DataFrame with questions and answers
        """
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train data not found: {self.train_path}")
        
        # Support CSV, JSON, JSONL formats
        if self.train_path.endswith('.csv'):
            return pd.read_csv(self.train_path)
        elif self.train_path.endswith('.json'):
            return pd.read_json(self.train_path)
        elif self.train_path.endswith('.jsonl'):
            return pd.read_json(self.train_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {self.train_path}")
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test Q&A data (questions only).
        
        Returns:
            DataFrame with questions
        """
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test data not found: {self.test_path}")
        
        # Support CSV, JSON, JSONL formats
        if self.test_path.endswith('.csv'):
            return pd.read_csv(self.test_path)
        elif self.test_path.endswith('.json'):
            return pd.read_json(self.test_path)
        elif self.test_path.endswith('.jsonl'):
            return pd.read_json(self.test_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {self.test_path}")
    
    def get_questions_and_answers(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Extract questions and answers from DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (questions, answers)
        """
        # Assumes columns named 'question' and 'answer'
        # Adjust based on actual competition data format
        questions = df['question'].tolist()
        answers = df.get('answer', [None] * len(questions)).tolist()
        
        return questions, answers
