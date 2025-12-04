"""
Run End-to-End Pipeline Script
Executes the complete RAG pipeline from data to submission.

Usage:
    python scripts/run_pipeline.py
"""

import sys
import os
from pathlib import Path
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging
import time


def run_command(command, description):
    """
    Run a shell command and log output.
    
    Args:
        command: Command to run
        description: Description of the step
    """
    logger = setup_logging(get_config())
    
    logger.info("\n" + "=" * 60)
    logger.info(description)
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ {description} completed in {elapsed:.2f} seconds")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ {description} failed!")
        logger.error(f"Error: {e}")
        return False


def main():
    """Main function to run complete pipeline."""
    config = get_config()
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("RUNNING COMPLETE RAG PIPELINE")
    logger.info("=" * 60)
    
    pipeline_start = time.time()
    
    # Step 1: Download data (optional, comment out if data already exists)
    # if not run_command(
    #     "python scripts/download_data.py",
    #     "STEP 1: DOWNLOADING KAGGLE DATA"
    # ):
    #     logger.error("Pipeline failed at data download step")
    #     sys.exit(1)
    
    # Step 2: Build index
    if not run_command(
        "python scripts/build_index.py",
        "STEP 1: BUILDING SEARCH INDEX"
    ):
        logger.error("Pipeline failed at index building step")
        sys.exit(1)
    
    # Step 3: (Optional) Evaluate on training set
    evaluate = input("\nDo you want to evaluate on training set? (y/n): ")
    if evaluate.lower() == 'y':
        if not run_command(
            "python scripts/evaluate.py",
            "STEP 2: EVALUATING ON TRAINING SET"
        ):
            logger.warning("Evaluation failed, but continuing...")
    
    # Step 4: Generate predictions
    if not run_command(
        "python scripts/generate_predictions.py",
        "STEP 3: GENERATING PREDICTIONS"
    ):
        logger.error("Pipeline failed at prediction generation step")
        sys.exit(1)
    
    # Summary
    total_time = time.time() - pipeline_start
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("\nNext steps:")
    logger.info("1. Check the submission file in data/submissions/")
    logger.info("2. Submit to Kaggle competition")
    logger.info("3. Review evaluation results (if ran)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
