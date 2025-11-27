import pandas as pd
from prophet import Prophet
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def train_predict_prophet(df: pd.DataFrame, horizon: int = 30) -> Dict[str, Any]:
    """
    Trains a Prophet model and generates forecasts.
    
    Args:
        df: DataFrame with 'timestamp' and 'value' columns.
        horizon: Number of days to forecast.
        
    Returns:
        Dictionary containing the forecast DataFrame and the model.
    """
    # Prepare data for Prophet (ds, y)
    df_prophet = df.rename(columns={'timestamp': 'ds', 'value': 'y'})
    
    # Aggregate to daily if not already (summing values)
    df_prophet = df_prophet.groupby('ds')['y'].sum().reset_index()
    
    # Remove timezone if present (Prophet doesn't support it)
    if df_prophet['ds'].dt.tz is not None:
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='US')
    
    logger.info("Training Prophet model...")
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    
    logger.info("Forecasting completed.")
    return {
        'forecast': forecast,
        'model': model
    }
