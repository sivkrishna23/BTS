import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def tune_xgboost(df: pd.DataFrame, target_col: str = 'value', 
                 feature_cols: List[str] = None, n_trials: int = 50) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using Optuna.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        feature_cols: List of feature columns
        n_trials: Number of Optuna trials
        
    Returns:
        Dictionary with best params and model
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        return rmse
    
    logger.info("Starting XGBoost hyperparameter tuning...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best RMSE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Train final model with best params
    best_model = XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    return {
        'model': best_model,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }

def tune_lightgbm(df: pd.DataFrame, target_col: str = 'value', 
                  feature_cols: List[str] = None, n_trials: int = 50) -> Dict[str, Any]:
    """
    Tune LightGBM hyperparameters using Optuna.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        feature_cols: List of feature columns
        n_trials: Number of Optuna trials
        
    Returns:
        Dictionary with best params and model
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        return rmse
    
    logger.info("Starting LightGBM hyperparameter tuning...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best RMSE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Train final model with best params
    best_model = LGBMRegressor(**study.best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    return {
        'model': best_model,
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }
