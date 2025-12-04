"""
Download Kaggle Data Script
Downloads competition data from Kaggle.

Usage:
    python scripts/download_data.py
    
Make sure to set KAGGLE_USERNAME and KAGGLE_KEY in .env file
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging


def download_kaggle_data():
    """Download data from Kaggle competition."""
    config = get_config()
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("DOWNLOADING KAGGLE DATA")
    logger.info("=" * 60)
    
    # Competition name (update with actual competition name)
    competition_name = "casml-generative-ai-hackathon"
    
    # Data directory
    data_dir = config.get('paths.raw_data')
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"\nDownloading data from competition: {competition_name}")
        logger.info(f"Saving to: {data_dir}")
        
        # Download all competition files
        api.competition_download_files(competition_name, path=data_dir)
        
        logger.info("\nDownload complete!")
        logger.info("Extracting files...")
        
        # Extract zip files
        import zipfile
        for file in os.listdir(data_dir):
            if file.endswith('.zip'):
                zip_path = os.path.join(data_dir, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                logger.info(f"Extracted: {file}")
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA DOWNLOAD COMPLETE")
        logger.info(f"Files saved to: {data_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\nError downloading data: {e}")
        logger.error("\nMake sure you have:")
        logger.error("1. Accepted the competition rules on Kaggle")
        logger.error("2. Set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
        logger.error("3. Installed kaggle package: pip install kaggle")
        sys.exit(1)


if __name__ == "__main__":
    download_kaggle_data()
