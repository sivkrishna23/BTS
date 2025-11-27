import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_timestamps(df: pd.DataFrame, date_col: str = 'date', timezone: str = 'UTC') -> pd.DataFrame:
    """
    Convert a column to datetime and handle timezone.
    
    Args:
        df: Input DataFrame.
        date_col: Name of the column containing date/time information.
        timezone: Target timezone.
        
    Returns:
        pd.DataFrame: DataFrame with a 'timestamp' column in the specified timezone.
    """
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")
    
    try:
        df['timestamp'] = pd.to_datetime(df[date_col])
        # If naive, localize to UTC then convert. If aware, convert.
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC') # Assuming UTC if not specified
        
        df['timestamp'] = df['timestamp'].dt.tz_convert(timezone)
        logger.info(f"Converted '{date_col}' to timestamp with timezone {timezone}")
        return df
    except Exception as e:
        logger.error(f"Error cleaning timestamps: {e}")
        raise

def clean_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a column to numeric, handling commas and errors.
    """
    df = df.copy()
    if col in df.columns:
        # Remove commas if present (assuming string)
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def handle_duplicates(df: pd.DataFrame, subset: list = None, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows.
    
    Args:
        df: Input DataFrame.
        subset: Column label or sequence of labels to consider for identifying duplicates.
        keep: 'first', 'last', or False.
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    df = df.copy()
    initial_shape = df.shape
    df = df.drop_duplicates(subset=subset, keep=keep)
    final_shape = df.shape
    
    if initial_shape != final_shape:
        logger.info(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    
    return df

def impute_missing(df: pd.DataFrame, method: str = 'linear', limit: int = None) -> pd.DataFrame:
    """
    Impute missing values in numeric columns.
    
    Args:
        df: Input DataFrame.
        method: Interpolation method ('linear', 'time', etc.) or 'ffill', 'bfill'.
        limit: Maximum number of consecutive NaNs to fill.
        
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Check for missing values
    missing_counts = df[numeric_cols].isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
        
        if method in ['ffill', 'bfill']:
            df[numeric_cols] = df[numeric_cols].fillna(method=method, limit=limit)
        else:
            # For interpolation, we need a datetime index usually
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp').sort_index()
                df[numeric_cols] = df[numeric_cols].interpolate(method=method, limit=limit)
                df = df.reset_index()
            else:
                 df[numeric_cols] = df[numeric_cols].interpolate(method=method, limit=limit)
                 
        logger.info(f"Imputed missing values using method: {method}")
    
    return df
