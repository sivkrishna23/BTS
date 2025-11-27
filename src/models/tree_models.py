import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def train_xgboost(df: pd.DataFrame, target_col: str = 'value', feature_cols: List[str] = None) -> Dict[str, Any]:
    """Train XGBoost model."""
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'y_train': y_train,
        'y_test': y_test,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }

def train_lightgbm(df: pd.DataFrame, target_col: str = 'value', feature_cols: List[str] = None) -> Dict[str, Any]:
    """Train LightGBM model."""
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LGBMRegressor(
        n_estimators=500,
        num_leaves=128,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
    
    logger.info("Training LightGBM model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'y_train': y_train,
        'y_test': y_test,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }
