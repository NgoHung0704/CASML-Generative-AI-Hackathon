"""
Generate Predictions Script
Runs RAG pipeline on test set and generates predictions.

Usage:
    python scripts/generate_predictions.py
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, set_seed, get_timestamp, configure_gpu
from src.ingestion import QADataLoader
from src.indexing import EmbeddingGenerator, IndexBuilder
from src.retrieval import Retriever
from src.generation import LLMGenerator, RAGPipeline
from src.evaluation import SubmissionGenerator
import time


def main():
    """Main function to generate predictions."""
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
    logger.info("GENERATING PREDICTIONS")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load indexes
    logger.info("\n[1/5] Loading search indexes...")
    embedding_generator = EmbeddingGenerator(config)
    index_builder = IndexBuilder(config)
    
    indexes_dir = config.get('paths.indexes_dir')
    index_builder.load_index(indexes_dir)
    logger.info(f"Loaded indexes from: {indexes_dir}")
    
    # Step 2: Initialize retriever
    logger.info("\n[2/5] Initializing retriever...")
    retriever = Retriever(config, embedding_generator, index_builder)
    logger.info(f"Retrieval strategy: {config.get('retrieval.strategy')}")
    logger.info(f"Top-K: {config.get('retrieval.top_k')}")
    
    # Step 3: Initialize generator
    logger.info("\n[3/5] Initializing LLM generator...")
    generator = LLMGenerator(config)
    logger.info(f"Model: {config.get('generation.model.model_name')}")
    
    # Step 4: Create RAG pipeline
    logger.info("\n[4/5] Creating RAG pipeline...")
    rag_pipeline = RAGPipeline(config, retriever, generator)
    
    # Step 5: Load test data and generate predictions
    logger.info("\n[5/5] Generating predictions for test set...")
    qa_loader = QADataLoader(config)
    test_df = qa_loader.load_test_data()
    
    logger.info(f"Test set size: {len(test_df)}")
    
    predictions = []
    question_ids = []
    
    for idx, row in test_df.iterrows():
        # Assumes columns: 'id', 'question'
        # Adjust based on actual competition data format
        question_id = row.get('id', idx)
        question = row['question']
        
        logger.info(f"\nProcessing question {idx+1}/{len(test_df)}: {question[:100]}...")
        
        # Generate answer
        result = rag_pipeline.answer_question(question)
        answer = result['answer']
        
        predictions.append(answer)
        question_ids.append(question_id)
        
        logger.info(f"Answer: {answer[:100]}...")
    
    # Generate submission file
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING SUBMISSION FILE")
    logger.info("=" * 60)
    
    submission_gen = SubmissionGenerator(config)
    timestamp = get_timestamp()
    submission_path = submission_gen.generate_submission(
        question_ids,
        predictions,
        output_filename=f'submission_{timestamp}.csv'
    )
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION GENERATION COMPLETE")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Questions processed: {len(predictions)}")
    logger.info(f"Submission file: {submission_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
