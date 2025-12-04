"""
Evaluate Model Script
Evaluates model predictions against ground truth.

Usage:
    python scripts/evaluate.py
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, set_seed, configure_gpu
from src.ingestion import QADataLoader
from src.indexing import EmbeddingGenerator, IndexBuilder
from src.retrieval import Retriever
from src.generation import LLMGenerator, RAGPipeline
from src.evaluation import Evaluator
import time


def main():
    """Main function to evaluate model."""
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
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load indexes
    logger.info("\n[1/6] Loading search indexes...")
    embedding_generator = EmbeddingGenerator(config)
    index_builder = IndexBuilder(config)
    
    indexes_dir = config.get('paths.indexes_dir')
    index_builder.load_index(indexes_dir)
    logger.info(f"Loaded indexes from: {indexes_dir}")
    
    # Step 2: Initialize retriever
    logger.info("\n[2/6] Initializing retriever...")
    retriever = Retriever(config, embedding_generator, index_builder)
    
    # Step 3: Initialize generator
    logger.info("\n[3/6] Initializing LLM generator...")
    generator = LLMGenerator(config)
    
    # Step 4: Create RAG pipeline
    logger.info("\n[4/6] Creating RAG pipeline...")
    rag_pipeline = RAGPipeline(config, retriever, generator)
    
    # Step 5: Load training data
    logger.info("\n[5/6] Loading training data...")
    qa_loader = QADataLoader(config)
    train_df = qa_loader.load_train_data()
    
    questions, ground_truths = qa_loader.get_questions_and_answers(train_df)
    logger.info(f"Loaded {len(questions)} Q&A pairs")
    
    # Step 6: Generate predictions and evaluate
    logger.info("\n[6/6] Generating predictions and evaluating...")
    
    predictions = []
    
    for idx, question in enumerate(questions):
        logger.info(f"\nProcessing {idx+1}/{len(questions)}: {question[:100]}...")
        
        result = rag_pipeline.answer_question(question)
        answer = result['answer']
        predictions.append(answer)
        
        logger.info(f"Predicted: {answer[:100]}...")
        logger.info(f"Ground truth: {ground_truths[idx][:100]}...")
    
    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 60)
    
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(predictions, ground_truths)
    
    logger.info("\nEvaluation Results:")
    for metric_name, score in metrics.items():
        if isinstance(score, list):
            logger.info(f"  {metric_name}: {score}")
        else:
            logger.info(f"  {metric_name}: {score:.4f}")
    
    # Save results
    from src.utils import save_results, get_timestamp
    
    results = {
        'metrics': metrics,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'questions': questions
    }
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = get_timestamp()
    results_path = os.path.join(results_dir, f'evaluation_{timestamp}.json')
    save_results(results, results_path)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Questions evaluated: {len(predictions)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
