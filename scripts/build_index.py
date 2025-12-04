"""
Build Index Script
Creates embeddings and search indexes from corpus.

Usage:
    python scripts/build_index.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, set_seed, configure_gpu
from src.ingestion import CorpusLoader
from src.indexing import TextChunker, EmbeddingGenerator, IndexBuilder
import time


def main():
    """Main function to build indexes."""
    # Load configuration
    config = get_config()
    
    # Setup logging and seed
    logger = setup_logging(config)
    set_seed(config.get('project.seed', 42))
    
    # Configure TensorFlow GPU
    gpu_memory_growth = config.get('resources.gpu_memory_growth', True)
    gpu_memory_limit = config.get('resources.gpu_memory_limit', None)
    configure_gpu(gpu_memory_growth, gpu_memory_limit)
    
    logger.info("=" * 60)
    logger.info("BUILDING SEARCH INDEX")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load corpus
    logger.info("\n[1/4] Loading corpus...")
    corpus_loader = CorpusLoader(config)
    corpus = corpus_loader.load_and_preprocess()
    logger.info(f"Loaded corpus: {len(corpus)} characters")
    
    # Step 2: Chunk text
    logger.info("\n[2/4] Chunking text...")
    chunker = TextChunker(config)
    chunks = chunker.chunk_text(corpus)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Save chunks
    chunks_dir = config.get('paths.processed_data')
    os.makedirs(chunks_dir, exist_ok=True)
    
    import pickle
    chunks_path = os.path.join(chunks_dir, 'chunks.pkl')
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved chunks to: {chunks_path}")
    
    # Step 3: Generate embeddings
    logger.info("\n[3/4] Generating embeddings...")
    embedding_generator = EmbeddingGenerator(config)
    embeddings = embedding_generator.encode(chunks, show_progress=True)
    logger.info(f"Generated embeddings: {embeddings.shape}")
    
    # Save embeddings
    embeddings_dir = config.get('paths.embeddings_dir')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
    embedding_generator.save_embeddings(embeddings, embeddings_path)
    logger.info(f"Saved embeddings to: {embeddings_path}")
    
    # Step 4: Build indexes
    logger.info("\n[4/4] Building search indexes...")
    index_builder = IndexBuilder(config)
    
    index_type = config.get('indexing.index.type', 'faiss')
    if index_type == 'faiss':
        index_builder.build_faiss_index(embeddings)
    elif index_type == 'bm25':
        index_builder.build_bm25_index(chunks)
    elif index_type == 'hybrid':
        index_builder.build_hybrid_index(embeddings, chunks)
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    logger.info(f"Built {index_type} index")
    
    # Save indexes
    indexes_dir = config.get('paths.indexes_dir')
    index_builder.save_index(indexes_dir)
    logger.info(f"Saved indexes to: {indexes_dir}")
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Chunks: {len(chunks)}")
    logger.info(f"Embeddings: {embeddings.shape}")
    logger.info(f"Index type: {index_type}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
