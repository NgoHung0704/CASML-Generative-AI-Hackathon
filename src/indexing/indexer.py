"""
Indexing module - Text chunking, embedding generation, and index building
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import tensorflow as tf


class TextChunker:
    """
    Splits corpus into chunks for indexing.
    """
    
    def __init__(self, config):
        """
        Initialize text chunker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.strategy = config.get('indexing.chunking.strategy', 'fixed')
        self.chunk_size = config.get('indexing.chunking.chunk_size', 512)
        self.chunk_overlap = config.get('indexing.chunking.chunk_overlap', 50)
        self.separator = config.get('indexing.chunking.separator', '\n\n')
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on configured strategy.
        
        Args:
            text: Input text
        
        Returns:
            List of text chunks
        """
        if self.strategy == 'fixed':
            return self._chunk_fixed_size(text)
        elif self.strategy == 'semantic':
            return self._chunk_semantic(text)
        elif self.strategy == 'sentence':
            return self._chunk_by_sentence(text)
        elif self.strategy == 'paragraph':
            return self._chunk_by_paragraph(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Split by fixed token/character count with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[str]:
        """Split by semantic boundaries (paragraphs with size limit)"""
        paragraphs = text.split(self.separator)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Split by sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
    
    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Split by paragraphs"""
        paragraphs = text.split(self.separator)
        return [p.strip() for p in paragraphs if p.strip()]


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using sentence transformers with TensorFlow backend.
    """
    
    def __init__(self, config):
        """
        Initialize embedding generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_name = config.get('indexing.embedding.model_name')
        self.device = config.get('indexing.embedding.device', 'CPU')
        self.batch_size = config.get('indexing.embedding.batch_size', 32)
        self.max_seq_length = config.get('indexing.embedding.max_seq_length', 512)
        self.normalize = config.get('indexing.embedding.normalize_embeddings', True)
        
        # Configure TensorFlow GPU
        if 'GPU' in self.device:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
        
        # Load model with TensorFlow backend
        # SentenceTransformers will automatically use TensorFlow if PyTorch is not available
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = self.max_seq_length
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
        
        Returns:
            Numpy array of embeddings (N x D)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """Save embeddings to disk"""
        np.save(output_path, embeddings)
    
    def load_embeddings(self, input_path: str) -> np.ndarray:
        """Load embeddings from disk"""
        return np.load(input_path)


class IndexBuilder:
    """
    Builds search indexes (FAISS, BM25, or hybrid).
    """
    
    def __init__(self, config):
        """
        Initialize index builder.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.index_type = config.get('indexing.index.type', 'faiss')
        self.faiss_index_type = config.get('indexing.index.faiss_index_type', 'IndexFlatIP')
        
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = None
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Document embeddings (N x D)
        """
        dimension = embeddings.shape[1]
        
        if self.faiss_index_type == 'IndexFlatIP':
            self.faiss_index = faiss.IndexFlatIP(dimension)
        elif self.faiss_index_type == 'IndexFlatL2':
            self.faiss_index = faiss.IndexFlatL2(dimension)
        elif self.faiss_index_type == 'IndexIVFFlat':
            nlist = self.config.get('indexing.index.faiss_nlist', 100)
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.train(embeddings)
        else:
            raise ValueError(f"Unknown FAISS index type: {self.faiss_index_type}")
        
        self.faiss_index.add(embeddings)
    
    def build_bm25_index(self, chunks: List[str]):
        """
        Build BM25 index from text chunks.
        
        Args:
            chunks: List of text chunks
        """
        # Tokenize chunks for BM25
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_chunks)
        self.chunks = chunks
    
    def build_hybrid_index(self, embeddings: np.ndarray, chunks: List[str]):
        """
        Build both FAISS and BM25 indexes.
        
        Args:
            embeddings: Document embeddings
            chunks: Text chunks
        """
        self.build_faiss_index(embeddings)
        self.build_bm25_index(chunks)
    
    def save_index(self, output_dir: str):
        """
        Save indexes to disk.
        
        Args:
            output_dir: Directory to save indexes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(output_dir, 'faiss.index'))
        
        if self.bm25_index is not None:
            with open(os.path.join(output_dir, 'bm25.pkl'), 'wb') as f:
                pickle.dump(self.bm25_index, f)
        
        if self.chunks is not None:
            with open(os.path.join(output_dir, 'chunks.pkl'), 'wb') as f:
                pickle.dump(self.chunks, f)
    
    def load_index(self, input_dir: str):
        """
        Load indexes from disk.
        
        Args:
            input_dir: Directory containing indexes
        """
        faiss_path = os.path.join(input_dir, 'faiss.index')
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
        
        bm25_path = os.path.join(input_dir, 'bm25.pkl')
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
        
        chunks_path = os.path.join(input_dir, 'chunks.pkl')
        if os.path.exists(chunks_path):
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
