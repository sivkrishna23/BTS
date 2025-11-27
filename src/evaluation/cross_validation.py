import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)

def rolling_origin_cv(df: pd.DataFrame, 
                     train_func: Callable,
                     n_splits: int = 5,
                     test_size: int = 30,
                     gap: int = 0) -> List[Dict[str, Any]]:
    """
    Perform rolling origin cross-validation for time series.
    
    Args:
        df: Time series DataFrame
        train_func: Function that takes train_df and returns predictions
        n_splits: Number of CV splits
        test_size: Size of test set in each split
        gap: Gap between train and test (to avoid leakage)
        
    Returns:
        List of results for each fold
    """
    results = []
    total_size = len(df)
    
    # Calculate split points
    min_train_size = total_size - (n_splits * test_size) - (n_splits * gap)
    
    for i in range(n_splits):
        # Define train/test indices
        test_end = total_size - (i * test_size)
        test_start = test_end - test_size
        train_end = test_start - gap
        
        if train_end < min_train_size:
            logger.warning(f"Insufficient data for split {i+1}")
            continue
        
        # Split data
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        logger.info(f"Fold {i+1}/{n_splits}: Train size={len(train_df)}, Test size={len(test_df)}")
        
        # Train and evaluate
        fold_result = train_func(train_df, test_df)
        fold_result['fold'] = i + 1
        results.append(fold_result)
    
    return results

def calculate_cv_metrics(cv_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average metrics across CV folds.
    
    Args:
        cv_results: List of results from rolling_origin_cv
        
    Returns:
        Dictionary of averaged metrics
    """
    from src.evaluation.metrics import calculate_metrics
    
    all_metrics = []
    for result in cv_results:
        if 'y_true' in result and 'y_pred' in result:
            metrics = calculate_metrics(result['y_true'], result['y_pred'])
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[f'{key}_mean'] = np.mean([m[key] for m in all_metrics])
            avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
    
    return avg_metrics
