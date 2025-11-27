import pandas as pd
import numpy as np
from typing import List

def create_lag_features(df: pd.DataFrame, target_col: str = 'value', lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
    """Create lag features for time series."""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df: pd.DataFrame, target_col: str = 'value', windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """Create rolling window statistics."""
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df

def create_date_features(df: pd.DataFrame, date_col: str = 'timestamp') -> pd.DataFrame:
    """Create date-based features."""
    df = df.copy()
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def prepare_ml_features(df: pd.DataFrame, target_col: str = 'value') -> pd.DataFrame:
    """Prepare all features for ML models."""
    df = df.copy()
    df = create_date_features(df)
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    
    # Drop rows with NaN (from lag/rolling features)
    df = df.dropna()
    
    return df
