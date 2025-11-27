import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)

def load_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV or Excel file.
    
    Args:
        filepath: Path to the file.
        **kwargs: Additional arguments passed to pd.read_csv or pd.read_excel.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, **kwargs)
        elif filepath.suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Successfully loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """
    df = df.copy()
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('-', '_')
                  .str.replace('/', '_')
                  .str.replace('.', '')
                  .str.replace('(', '')
                  .str.replace(')', ''))
    return df
