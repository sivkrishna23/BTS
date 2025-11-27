"""
Border Traffic System (BTS) - Standalone Data Analytics Pipeline
==================================================================
Complete data analytics pipeline from data loading to port-wise forecasting.

Usage:
    python bts_analytics.py --data_path <path_to_csv> --output_dir <output_directory>

Author: BTS Analytics Team
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
import argparse
import json

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Statistical Analysis
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# Machine Learning Models
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DataLoader:
    """Handle data loading and initial preprocessing."""
    
    @staticmethod
    def load_data(filepath):
        """Load data from CSV or Excel file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        print(f"✅ Loaded {len(df):,} rows from {filepath.name}")
        return df
    
    @staticmethod
    def standardize_columns(df):
        """Standardize column names to snake_case."""
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
    
    @staticmethod
    def clean_timestamps(df, date_col='date'):
        """Convert date column to datetime."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.rename(columns={date_col: 'timestamp'})
        return df
    
    @staticmethod
    def clean_numeric(df, col='value'):
        """Clean numeric column."""
        df = df.copy()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[col])
        df = df[df[col] >= 0]  # Remove negative values
        return df
    
    @staticmethod
    def handle_duplicates(df):
        """Remove duplicate rows."""
        initial_count = len(df)
        df = df.drop_duplicates()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"🔧 Removed {removed:,} duplicate rows")
        return df
    
    @staticmethod
    def impute_missing(df):
        """Handle missing values."""
        df = df.copy()
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"🔧 Handling {missing_count:,} missing values")
            df = df.fillna(method='ffill').fillna(method='bfill')
        return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Create features for machine learning models."""
    
    @staticmethod
    def prepare_ml_features(df):
        """Create time-based features for ML models."""
        df = df.copy()
        
        # Temporal features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['quarter'] = df['timestamp'].dt.quarter
        df['weekofyear'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # Lag features
        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        
        # Drop rows with NaN from lag/rolling features
        df = df.dropna()
        
        return df


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

class EDAAnalyzer:
    """Perform exploratory data analysis."""
    
    @staticmethod
    def statistical_summary(df):
        """Generate statistical summary."""
        stats_dict = {
            'Count': len(df),
            'Mean': df['value'].mean(),
            'Median': df['value'].median(),
            'Std Dev': df['value'].std(),
            'Min': df['value'].min(),
            'Max': df['value'].max(),
            'Range': df['value'].max() - df['value'].min(),
            'Skewness': df['value'].skew(),
            'Kurtosis': df['value'].kurtosis(),
            'CV (%)': (df['value'].std() / df['value'].mean()) * 100
        }
        
        print("\n📊 Statistical Summary:")
        print("=" * 50)
        for key, value in stats_dict.items():
            print(f"{key:15s}: {value:,.2f}")
        print("=" * 50)
        
        return stats_dict
    
    @staticmethod
    def temporal_analysis(df):
        """Analyze temporal patterns."""
        df = df.copy()
        
        # By day of week
        df['day_name'] = df['timestamp'].dt.day_name()
        day_avg = df.groupby('day_name')['value'].mean()
        
        # By month
        df['month_name'] = df['timestamp'].dt.month_name()
        month_avg = df.groupby('month_name')['value'].mean()
        
        print("\n📅 Average Traffic by Day of Week:")
        print(day_avg.sort_values(ascending=False))
        
        print("\n📅 Average Traffic by Month:")
        print(month_avg.sort_values(ascending=False))
        
        return {'day_avg': day_avg, 'month_avg': month_avg}
    
    @staticmethod
    def create_visualizations(df, output_dir):
        """Create and save visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines',
            name='Traffic Volume',
            line=dict(color='#00D9FF', width=2)
        ))
        fig.update_layout(
            title='Border Crossing Traffic Over Time',
            xaxis_title='Date',
            yaxis_title='Traffic Volume',
            template='plotly_dark',
            height=500
        )
        fig.write_html(output_dir / 'timeseries.html')
        print(f"📈 Saved: {output_dir / 'timeseries.html'}")
        
        # Distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['value'],
            nbinsx=50,
            name='Distribution',
            marker_color='#FF2E63'
        ))
        fig.update_layout(
            title='Traffic Volume Distribution',
            xaxis_title='Traffic Volume',
            yaxis_title='Frequency',
            template='plotly_dark',
            height=400
        )
        fig.write_html(output_dir / 'distribution.html')
        print(f"📊 Saved: {output_dir / 'distribution.html'}")


# ============================================================================
# FORECASTING MODELS
# ============================================================================

class ForecastingEngine:
    """Train and evaluate forecasting models."""
    
    @staticmethod
    def train_prophet(train_data, periods):
        """Train Prophet model."""
        print("\n🔮 Training Prophet model...")
        
        # Prepare data for Prophet
        prophet_df = train_data[['timestamp', 'value']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        # Make predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        print("✅ Prophet training complete")
        return {'model': model, 'forecast': forecast}
    
    @staticmethod
    def train_sarima(train_data, test_data):
        """Train SARIMA model."""
        print("\n📈 Training SARIMA model...")
        
        try:
            model = SARIMAX(
                train_data['value'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(test_data))
            
            print("✅ SARIMA training complete")
            return {'model': fitted_model, 'forecast': forecast.values}
        except Exception as e:
            print(f"⚠️ SARIMA training failed: {e}")
            return None
    
    @staticmethod
    def train_xgboost(train_ml, test_ml, feature_cols):
        """Train XGBoost model."""
        print("\n🌲 Training XGBoost model...")
        
        model = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(train_ml[feature_cols], train_ml['value'])
        predictions = model.predict(test_ml[feature_cols])
        
        print("✅ XGBoost training complete")
        return {'model': model, 'predictions': predictions}
    
    @staticmethod
    def train_lightgbm(train_ml, test_ml, feature_cols):
        """Train LightGBM model."""
        print("\n💡 Training LightGBM model...")
        
        model = LGBMRegressor(
            n_estimators=500,
            num_leaves=128,
            learning_rate=0.03,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        model.fit(train_ml[feature_cols], train_ml['value'])
        predictions = model.predict(test_ml[feature_cols])
        
        print("✅ LightGBM training complete")
        return {'model': model, 'predictions': predictions}


# ============================================================================
# MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate performance metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Accuracy (%)': (1 - mape/100) * 100
        }
    
    @staticmethod
    def compare_models(results, y_true):
        """Compare all models."""
        print("\n🏆 Model Performance Comparison:")
        print("=" * 80)
        print(f"{'Model':<15} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10} {'Accuracy':<10}")
        print("=" * 80)
        
        best_model = None
        best_r2 = -np.inf
        
        for model_name, predictions in results.items():
            metrics = ModelEvaluator.calculate_metrics(y_true, predictions)
            print(f"{model_name:<15} {metrics['R²']:<10.4f} {metrics['RMSE']:<12.2f} "
                  f"{metrics['MAE']:<12.2f} {metrics['MAPE']:<10.2f}% {metrics['Accuracy (%)']:<10.2f}%")
            
            if metrics['R²'] > best_r2:
                best_r2 = metrics['R²']
                best_model = model_name
        
        print("=" * 80)
        print(f"🥇 Best Model: {best_model} (R² = {best_r2:.4f})")
        print("=" * 80)
        
        return best_model


# ============================================================================
# PORT-WISE FORECASTING
# ============================================================================

class PortWiseForecaster:
    """Generate port-wise forecasts."""
    
    @staticmethod
    def forecast_by_port(df, port_col, horizon_days=90, output_dir=None):
        """Generate forecasts for each port."""
        print(f"\n🔮 Generating {horizon_days}-day forecasts for all ports...")
        print("=" * 80)
        
        ports = df[port_col].unique()
        forecasts = {}
        
        for port in ports:
            print(f"\n📍 Processing: {port}")
            
            # Filter data for this port
            port_data = df[df[port_col] == port].copy()
            port_agg = port_data.groupby('timestamp')['value'].sum().reset_index()
            
            if len(port_agg) < 30:
                print(f"⚠️ Insufficient data for {port} (only {len(port_agg)} records)")
                continue
            
            # Train Prophet model
            prophet_df = port_agg[['timestamp', 'value']].copy()
            prophet_df.columns = ['ds', 'y']
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(prophet_df)
            
            # Generate forecast
            future = model.make_future_dataframe(periods=horizon_days)
            forecast = model.predict(future)
            
            # Extract future predictions
            future_forecast = forecast.tail(horizon_days)
            
            # Calculate statistics
            avg_forecast = future_forecast['yhat'].mean()
            max_forecast = future_forecast['yhat'].max()
            min_forecast = future_forecast['yhat'].min()
            historical_avg = port_agg['value'].mean()
            growth = ((avg_forecast - historical_avg) / historical_avg) * 100
            
            forecasts[port] = {
                'forecast': future_forecast,
                'avg_predicted': avg_forecast,
                'max_predicted': max_forecast,
                'min_predicted': min_forecast,
                'growth_pct': growth,
                'historical_avg': historical_avg
            }
            
            print(f"  ✅ Avg Forecast: {avg_forecast:,.0f}")
            print(f"  📈 Growth: {growth:+.1f}%")
            
            # Save visualization if output directory provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                fig = go.Figure()
                
                # Historical
                fig.add_trace(go.Scatter(
                    x=port_agg['timestamp'],
                    y=port_agg['value'],
                    name='Historical',
                    line=dict(color='#00D9FF', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat'],
                    name='Forecast',
                    line=dict(color='#FF2E63', width=3)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=pd.concat([future_forecast['ds'], future_forecast['ds'][::-1]]),
                    y=pd.concat([future_forecast['yhat_upper'], future_forecast['yhat_lower'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 46, 99, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f'Traffic Forecast: {port} (Next {horizon_days} Days)',
                    xaxis_title='Date',
                    yaxis_title='Traffic Volume',
                    template='plotly_dark',
                    height=600
                )
                
                safe_port_name = port.replace('/', '_').replace(' ', '_')
                fig.write_html(output_dir / f'forecast_{safe_port_name}.html')
        
        print("\n" + "=" * 80)
        print(f"✅ Completed forecasts for {len(forecasts)} ports")
        
        return forecasts
    
    @staticmethod
    def save_forecast_summary(forecasts, output_path):
        """Save forecast summary to CSV."""
        summary_data = []
        
        for port, data in forecasts.items():
            summary_data.append({
                'Port': port,
                'Historical Avg': data['historical_avg'],
                'Predicted Avg': data['avg_predicted'],
                'Predicted Max': data['max_predicted'],
                'Predicted Min': data['min_predicted'],
                'Growth (%)': data['growth_pct']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Growth (%)', ascending=False)
        
        output_path = Path(output_path)
        summary_df.to_csv(output_path, index=False)
        print(f"\n💾 Forecast summary saved to: {output_path}")
        
        return summary_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class BTSAnalyticsPipeline:
    """Main analytics pipeline orchestrator."""
    
    def __init__(self, data_path, output_dir='output'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.eda_analyzer = EDAAnalyzer()
        self.forecasting_engine = ForecastingEngine()
        self.evaluator = ModelEvaluator()
        self.port_forecaster = PortWiseForecaster()
    
    def run(self, forecast_horizon=90):
        """Execute complete analytics pipeline."""
        print("\n" + "=" * 80)
        print("🚀 BTS ANALYTICS PIPELINE STARTED")
        print("=" * 80)
        
        # Step 1: Load and preprocess data
        print("\n📂 STEP 1: Data Loading & Preprocessing")
        print("-" * 80)
        df_raw = self.loader.load_data(self.data_path)
        df = self.loader.standardize_columns(df_raw)
        
        if 'date' in df.columns:
            df = self.loader.clean_timestamps(df, date_col='date')
        if 'value' in df.columns:
            df = self.loader.clean_numeric(df, col='value')
        
        df = self.loader.handle_duplicates(df)
        df = self.loader.impute_missing(df)
        
        print(f"✅ Final dataset: {len(df):,} rows, {len(df.columns)} columns")
        
        # Step 2: Exploratory Data Analysis
        print("\n📊 STEP 2: Exploratory Data Analysis")
        print("-" * 80)
        self.eda_analyzer.statistical_summary(df.groupby('timestamp')['value'].sum().reset_index())
        self.eda_analyzer.temporal_analysis(df)
        self.eda_analyzer.create_visualizations(
            df.groupby('timestamp')['value'].sum().reset_index(),
            self.output_dir / 'eda'
        )
        
        # Step 3: Model Training & Evaluation (if test data available)
        print("\n🤖 STEP 3: Model Training & Evaluation")
        print("-" * 80)
        
        df_agg = df.groupby('timestamp')['value'].sum().reset_index()
        df_agg['year'] = df_agg['timestamp'].dt.year
        
        train_data = df_agg[df_agg['year'] <= 2024]
        test_data = df_agg[df_agg['year'] == 2025]
        
        if len(test_data) > 0:
            print(f"📊 Training: {len(train_data)} samples | Testing: {len(test_data)} samples")
            
            results = {}
            y_true = test_data['value'].values
            
            # Prophet
            prophet_result = self.forecasting_engine.train_prophet(train_data, len(test_data))
            results['Prophet'] = prophet_result['forecast'].tail(len(test_data))['yhat'].values
            
            # SARIMA
            sarima_result = self.forecasting_engine.train_sarima(train_data, test_data)
            if sarima_result:
                results['SARIMA'] = sarima_result['forecast']
            
            # ML Models
            df_ml = self.feature_engineer.prepare_ml_features(df_agg)
            train_ml = df_ml[df_ml['year'] <= 2024].drop('year', axis=1)
            test_ml = df_ml[df_ml['year'] == 2025].drop('year', axis=1)
            feature_cols = [col for col in train_ml.columns if col not in ['value', 'timestamp']]
            
            xgb_result = self.forecasting_engine.train_xgboost(train_ml, test_ml, feature_cols)
            results['XGBoost'] = xgb_result['predictions']
            
            lgbm_result = self.forecasting_engine.train_lightgbm(train_ml, test_ml, feature_cols)
            results['LightGBM'] = lgbm_result['predictions']
            
            # Compare models
            best_model = self.evaluator.compare_models(results, y_true)
        else:
            print("⚠️ No 2025 data available for testing. Skipping model evaluation.")
        
        # Step 4: Port-wise Forecasting
        print("\n🔮 STEP 4: Port-Wise Forecasting")
        print("-" * 80)
        
        port_col = next((col for col in df.columns if 'port' in col), None)
        
        if port_col:
            forecasts = self.port_forecaster.forecast_by_port(
                df, port_col, 
                horizon_days=forecast_horizon,
                output_dir=self.output_dir / 'port_forecasts'
            )
            
            summary_df = self.port_forecaster.save_forecast_summary(
                forecasts,
                self.output_dir / 'port_forecast_summary.csv'
            )
            
            print("\n📋 Top 5 Ports by Projected Growth:")
            print(summary_df.head()[['Port', 'Growth (%)']].to_string(index=False))
        else:
            print("⚠️ No port column found. Generating aggregate forecast only.")
            
            # Aggregate forecast
            prophet_result = self.forecasting_engine.train_prophet(df_agg, forecast_horizon)
            forecast = prophet_result['forecast']
            
            # Save aggregate forecast
            forecast.to_csv(self.output_dir / 'aggregate_forecast.csv', index=False)
            print(f"💾 Aggregate forecast saved to: {self.output_dir / 'aggregate_forecast.csv'}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ BTS ANALYTICS PIPELINE COMPLETED")
        print("=" * 80)
        print(f"\n📁 All outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.rglob('*')):
            if file.is_file():
                print(f"  - {file.relative_to(self.output_dir)}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='BTS Analytics Pipeline - Complete data analytics from loading to forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bts_analytics.py --data_path data/border_crossing.csv
  python bts_analytics.py --data_path data/border_crossing.csv --output_dir results --horizon 180
        """
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to input CSV or Excel file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output)'
    )
    
    parser.add_argument(
        '--horizon',
        type=int,
        default=90,
        help='Forecast horizon in days (default: 90)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = BTSAnalyticsPipeline(args.data_path, args.output_dir)
    pipeline.run(forecast_horizon=args.horizon)


if __name__ == '__main__':
    main()
