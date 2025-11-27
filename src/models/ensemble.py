import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def simple_average_ensemble(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Simple average ensemble of multiple model predictions.
    
    Args:
        predictions: Dictionary of model_name -> predictions array
        
    Returns:
        Averaged predictions
    """
    pred_arrays = list(predictions.values())
    return np.mean(pred_arrays, axis=0)

def weighted_average_ensemble(predictions: Dict[str, np.ndarray], 
                              weights: Dict[str, float]) -> np.ndarray:
    """
    Weighted average ensemble.
    
    Args:
        predictions: Dictionary of model_name -> predictions array
        weights: Dictionary of model_name -> weight
        
    Returns:
        Weighted averaged predictions
    """
    result = np.zeros_like(list(predictions.values())[0])
    total_weight = sum(weights.values())
    
    for model_name, preds in predictions.items():
        weight = weights.get(model_name, 1.0)
        result += preds * (weight / total_weight)
    
    return result

def train_stacking_ensemble(predictions: Dict[str, np.ndarray], 
                            y_true: np.ndarray) -> Dict[str, Any]:
    """
    Train a stacking ensemble using Ridge regression as meta-learner.
    
    Args:
        predictions: Dictionary of model_name -> predictions array (training set)
        y_true: True values
        
    Returns:
        Dictionary with meta-model and performance
    """
    # Create feature matrix from predictions
    X_meta = np.column_stack(list(predictions.values()))
    
    # Split for meta-model training
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y_true, test_size=0.2, shuffle=False
    )
    
    # Train meta-learner
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = meta_model.predict(X_train)
    test_pred = meta_model.predict(X_test)
    
    logger.info("Stacking ensemble trained successfully")
    
    return {
        'meta_model': meta_model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(predictions.keys()),
        'coefficients': dict(zip(predictions.keys(), meta_model.coef_))
    }

def predict_stacking_ensemble(meta_model, predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Make predictions using trained stacking ensemble.
    
    Args:
        meta_model: Trained meta-learner
        predictions: Dictionary of model_name -> predictions array
        
    Returns:
        Ensemble predictions
    """
    X_meta = np.column_stack(list(predictions.values()))
    return meta_model.predict(X_meta)
