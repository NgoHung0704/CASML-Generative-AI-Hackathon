"""
Retrieval module - Query processing and document retrieval
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder


class Retriever:
    """
    Handles query processing and document retrieval from indexes.
    """
    
    def __init__(self, config, embedding_generator, index_builder):
        """
        Initialize retriever.
        
        Args:
            config: Configuration object
            embedding_generator: EmbeddingGenerator instance
            index_builder: IndexBuilder instance
        """
        self.config = config
        self.embedding_generator = embedding_generator
        self.index_builder = index_builder
        
        self.strategy = config.get('retrieval.strategy', 'dense')
        self.top_k = config.get('retrieval.top_k', 5)
        self.dense_weight = config.get('retrieval.dense_weight', 0.6)
        self.sparse_weight = config.get('retrieval.sparse_weight', 0.4)
        
        # Re-ranker setup
        self.use_reranker = config.get('retrieval.use_reranker', False)
        self.rerank_top_k = config.get('retrieval.rerank_top_k', 3)
        
        if self.use_reranker:
            reranker_model = config.get('retrieval.reranker_model')
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None
    
    def retrieve(self, query: str) -> List[Dict[str, any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
        
        Returns:
            List of retrieved documents with scores
        """
        if self.strategy == 'dense':
            results = self._dense_retrieval(query)
        elif self.strategy == 'sparse':
            results = self._sparse_retrieval(query)
        elif self.strategy == 'hybrid':
            results = self._hybrid_retrieval(query)
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.strategy}")
        
        # Apply re-ranking if enabled
        if self.use_reranker and len(results) > 0:
            results = self._rerank(query, results)
        
        return results[:self.rerank_top_k if self.use_reranker else self.top_k]
    
    def _dense_retrieval(self, query: str) -> List[Dict[str, any]]:
        """
        Dense retrieval using FAISS.
        
        Args:
            query: Query string
        
        Returns:
            List of {chunk, score, index} dicts
        """
        # Encode query
        query_embedding = self.embedding_generator.encode([query])[0]
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Search FAISS index
        scores, indices = self.index_builder.faiss_index.search(query_embedding, self.top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    'chunk': self.index_builder.chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def _sparse_retrieval(self, query: str) -> List[Dict[str, any]]:
        """
        Sparse retrieval using BM25.
        
        Args:
            query: Query string
        
        Returns:
            List of {chunk, score, index} dicts
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.index_builder.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.index_builder.chunks[idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })
        
        return results
    
    def _hybrid_retrieval(self, query: str) -> List[Dict[str, any]]:
        """
        Hybrid retrieval combining dense and sparse methods.
        
        Args:
            query: Query string
        
        Returns:
            List of {chunk, score, index} dicts
        """
        # Get results from both methods
        dense_results = self._dense_retrieval(query)
        sparse_results = self._sparse_retrieval(query)
        
        # Normalize scores
        dense_scores = [r['score'] for r in dense_results]
        sparse_scores = [r['score'] for r in sparse_results]
        
        if dense_scores:
            max_dense = max(dense_scores)
            if max_dense > 0:
                for r in dense_results:
                    r['score'] /= max_dense
        
        if sparse_scores:
            max_sparse = max(sparse_scores)
            if max_sparse > 0:
                for r in sparse_results:
                    r['score'] /= max_sparse
        
        # Merge and re-score
        merged = {}
        for r in dense_results:
            idx = r['index']
            merged[idx] = {
                'chunk': r['chunk'],
                'score': r['score'] * self.dense_weight,
                'index': idx
            }
        
        for r in sparse_results:
            idx = r['index']
            if idx in merged:
                merged[idx]['score'] += r['score'] * self.sparse_weight
            else:
                merged[idx] = {
                    'chunk': r['chunk'],
                    'score': r['score'] * self.sparse_weight,
                    'index': idx
                }
        
        # Sort by combined score
        results = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
        
        return results[:self.top_k]
    
    def _rerank(self, query: str, results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Query string
            results: Initial retrieval results
        
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [[query, r['chunk']] for r in results]
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores and sort
        for i, score in enumerate(rerank_scores):
            results[i]['rerank_score'] = float(score)
            results[i]['original_score'] = results[i]['score']
            results[i]['score'] = float(score)
        
        results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        return results
