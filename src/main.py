import logging
from pathlib import Path
import sys
import pandas as pd

# Add the project root to the python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.fetcher import fetch_data, DATA_URL, DESTINATION
from src.data.loader import load_data, standardize_columns
from src.data.cleaning import clean_timestamps, handle_duplicates, impute_missing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Border Crossing Traffic Forecasting Pipeline...")
    
    # 1. Data Ingestion
    data_path = Path("data/raw/Border_Crossing_Entry_Data.csv")
    if not data_path.exists():
        logger.info("Data file not found. Fetching data...")
        fetch_data(DATA_URL, data_path)
    
    logger.info("Loading data...")
    df = load_data(data_path)
    
    # 2. Data Cleaning
    logger.info("Cleaning data...")
    df = standardize_columns(df)
    
    # The dataset usually has a 'date' column
    if 'date' in df.columns:
        df = clean_timestamps(df, date_col='date')
    
    df = handle_duplicates(df)
    
    # 3. Save Processed Data
    processed_path = Path("data/processed/cleaned_data.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
