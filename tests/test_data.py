import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.loader import standardize_columns
from src.data.cleaning import clean_timestamps, handle_duplicates, impute_missing

def test_standardize_columns():
    df = pd.DataFrame({'Date Time': [], 'Port Name': [], 'Value (Count)': []})
    df = standardize_columns(df)
    assert 'date_time' in df.columns
    assert 'port_name' in df.columns
    assert 'value_count' in df.columns

def test_clean_timestamps():
    df = pd.DataFrame({'date': ['2023-01-01', '2023-01-02']})
    df = clean_timestamps(df, date_col='date')
    assert 'timestamp' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
    assert df['timestamp'].dt.tz is not None

def test_handle_duplicates():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
    df = handle_duplicates(df)
    assert len(df) == 2

def test_impute_missing():
    df = pd.DataFrame({'a': [1, np.nan, 3], 'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])})
    df = impute_missing(df, method='linear')
    assert df['a'].isnull().sum() == 0
    assert df.iloc[1]['a'] == 2.0
