import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def train_sarima(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                 order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) -> Dict[str, Any]:
    """
    Train SARIMA model.
    
    Args:
        train_df: Training data with 'value' column
        test_df: Test data with 'value' column
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
        
    Returns:
        Dictionary with model and predictions
    """
    logger.info("Training SARIMA model...")
    
    # Prepare data
    y_train = train_df['value'].values
    y_test = test_df['value'].values
    
    # Train SARIMA
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    
    # Predictions
    train_pred = fitted_model.fittedvalues
    test_pred = fitted_model.forecast(steps=len(y_test))
    
    logger.info("SARIMA model trained successfully")
    
    return {
        'model': fitted_model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'y_train': y_train,
        'y_test': y_test
    }
