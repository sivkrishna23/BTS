import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.loader import load_data, standardize_columns
from src.data.cleaning import clean_timestamps, handle_duplicates, impute_missing, clean_numeric
from src.features.engineering import prepare_ml_features
from src.models.baseline import train_predict_prophet
from src.models.tree_models import train_xgboost, train_lightgbm
from src.models.sarima import train_sarima
from src.evaluation.metrics import calculate_metrics
from src.styles import get_custom_css

# Page config
st.set_page_config(
    page_title="Border Crossing Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Apply Custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Title with gradient effect
st.markdown('<h1 class="gradient-text">🛂 Border Crossing Traffic Forecasting Platform</h1>', unsafe_allow_html=True)
st.markdown("### **AI-Powered Predictive Analytics | Train on ≤2024 → Test on 2025**")

# Global state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# Main Tab
tab1, tab2 = st.tabs(["📂 Data Upload & Processing", "🎯 Complete Model Comparison & Analytics"])

# ===== TAB 1: Data Upload =====
with tab1:
    st.header("📂 Data Upload & Processing")
    
    uploaded_file = st.file_uploader("Upload Border Crossing Data (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing data..."):
                temp_path = Path("data/raw") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                df_raw = load_data(temp_path)
                st.success(f"✅ Loaded {len(df_raw):,} rows")
                
                progress_bar = st.progress(0)
                df_clean = standardize_columns(df_raw)
                progress_bar.progress(33)
                
                if 'date' in df_clean.columns:
                    df_clean = clean_timestamps(df_clean, date_col='date')
                if 'value' in df_clean.columns:
                    df_clean = clean_numeric(df_clean, col='value')
                progress_bar.progress(66)
                
                df_clean = handle_duplicates(df_clean)
                df_clean = impute_missing(df_clean)
                progress_bar.progress(100)
                
                st.session_state.df = df_clean
                
                processed_path = Path("data/processed/cleaned_data.csv")
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                df_clean.to_csv(processed_path, index=False)
                
                st.success("✅ Data Pipeline Complete!")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            
    elif Path("data/processed/cleaned_data.csv").exists():
        st.info("ℹ️ Using existing processed data")
        if st.session_state.df is None:
            st.session_state.df = pd.read_csv("data/processed/cleaned_data.csv")
            if 'timestamp' in st.session_state.df.columns:
                st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['timestamp'])
            if 'value' in st.session_state.df.columns:
                st.session_state.df = clean_numeric(st.session_state.df, col='value')
    
    if st.session_state.df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Rows", f"{len(st.session_state.df):,}")
        col2.metric("📅 Date Range", f"{st.session_state.df['timestamp'].min().date()}")
        col3.metric("📅 To", f"{st.session_state.df['timestamp'].max().date()}")
        col4.metric("📈 Total Volume", f"{st.session_state.df['value'].sum()/1e6:.1f}M")

# ===== TAB 2: Complete Model Comparison =====
with tab2:
    st.header("🎯 Complete Model Comparison & Analytics")
    st.markdown("### Train on data ≤ December 2024 | Test on 2025 data")
    
    if st.session_state.df is not None:
        df_agg = st.session_state.df.groupby('timestamp')['value'].sum().reset_index()
        df_agg['year'] = pd.to_datetime(df_agg['timestamp']).dt.year
        
        # Split data
        train_data = df_agg[df_agg['year'] <= 2024]
        test_data = df_agg[df_agg['year'] == 2025]
        
        # Display split info
        col1, col2, col3 = st.columns(3)
        col1.metric("🎓 Training Data (≤2024)", f"{len(train_data):,} days", delta="Training Set")
        col2.metric("🧪 Test Data (2025)", f"{len(test_data):,} days", delta="Test Set")
        if len(train_data) + len(test_data) > 0:
            col3.metric("📊 Split Ratio", f"{len(train_data)/(len(train_data)+len(test_data))*100:.1f}%", delta=f"{len(test_data)/(len(train_data)+len(test_data))*100:.1f}%")
        
        if len(test_data) == 0:
            st.warning("⚠️ No 2025 data available. Please upload data that includes 2025.")
        else:
            if st.button("🚀 Train All Models & Generate Comparison", key="train_all"):
                with st.spinner("Training all models... This may take a few minutes"):
                    results_dict = {}
                    
                    # Prepare features for ML models
                    df_ml_full = prepare_ml_features(df_agg)
                    train_ml = df_ml_full[df_ml_full['year'] <= 2024].drop('year', axis=1)
                    test_ml = df_ml_full[df_ml_full['year'] == 2025].drop('year', axis=1)
                    
                    y_test_actual = test_data['value'].values
                    test_dates = test_data['timestamp'].values
                    
                    # 1. Prophet
                    st.write("📊 Training Prophet...")
                    train_prophet = train_data[['timestamp', 'value']].copy()
                    results_prophet = train_predict_prophet(train_prophet, horizon=len(test_data))
                    forecast = results_prophet['forecast']
                    prophet_pred = forecast.tail(len(test_data))['yhat'].values
                    if len(prophet_pred) > len(y_test_actual):
                        prophet_pred = prophet_pred[:len(y_test_actual)]
                    results_dict['Prophet'] = prophet_pred
                    
                    # 2. SARIMA
                    st.write("📈 Training SARIMA...")
                    try:
                        sarima_results = train_sarima(train_data, test_data)
                        results_dict['SARIMA'] = sarima_results['test_pred']
                    except Exception as e:
                        st.warning(f"SARIMA training failed: {e}")
                    
                    # 3. XGBoost
                    st.write("🌲 Training XGBoost...")
                    feature_cols = [col for col in train_ml.columns if col not in ['value', 'timestamp']]
                    X_train = train_ml[feature_cols]
                    y_train = train_ml['value']
                    X_test = test_ml[feature_cols]
                    
                    from xgboost import XGBRegressor
                    xgb_model = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42)
                    xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    results_dict['XGBoost'] = xgb_pred
                    
                    # 4. LightGBM
                    st.write("💡 Training LightGBM...")
                    from lightgbm import LGBMRegressor
                    lgbm_model = LGBMRegressor(n_estimators=500, num_leaves=128, learning_rate=0.03, subsample=0.8, random_state=42, verbose=-1)
                    lgbm_model.fit(X_train, y_train)
                    lgbm_pred = lgbm_model.predict(X_test)
                    results_dict['LightGBM'] = lgbm_pred
                    
                    # Store results
                    st.session_state.comparison_results = {
                        'actual': y_test_actual,
                        'dates': test_dates,
                        'predictions': results_dict,
                        'train_data': train_data,
                        'test_data': test_data
                    }
                    
                    st.success("✅ All models trained successfully!")
            
            # Display results
            if st.session_state.comparison_results is not None:
                results = st.session_state.comparison_results
                
                st.markdown("---")
                st.subheader("📊 Model Predictions vs Actual (2025 Test Data)")
                
                # Create comparison plot
                fig = go.Figure()
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=results['dates'], 
                    y=results['actual'], 
                    name='Actual',
                    mode='lines+markers',
                    line=dict(color='#FFD700', width=4),
                    marker=dict(size=6)
                ))
                
                # Model predictions
                colors = {'Prophet': '#FF6B6B', 'SARIMA': '#4ECDC4', 'XGBoost': '#45B7D1', 'LightGBM': '#96CEB4'}
                for model_name, preds in results['predictions'].items():
                    if len(preds) > 0:
                        fig.add_trace(go.Scatter(
                            x=results['dates'][:len(preds)], 
                            y=preds, 
                            name=model_name,
                            mode='lines',
                            line=dict(color=colors.get(model_name, '#888888'), width=2)
                        ))
                
                fig.update_layout(
                    title="Model Predictions vs Actual Values (2025)",
                    xaxis_title="Date",
                    yaxis_title="Traffic Volume",
                    hovermode='x unified',
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=12),
                    legend=dict(
                        bgcolor='rgba(255,255,255,0.1)',
                        bordercolor='rgba(255,255,255,0.2)',
                        borderwidth=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics comparison
                st.markdown("---")
                st.subheader("📈 Model Performance Metrics (2025 Test Set)")
                
                metrics_data = []
                for model_name, preds in results['predictions'].items():
                    min_len = min(len(results['actual']), len(preds))
                    if min_len > 0:
                        metrics = calculate_metrics(results['actual'][:min_len], preds[:min_len])
                        metrics['Model'] = model_name
                        metrics['Accuracy (%)'] = (1 - metrics['MAPE']/100) * 100
                        metrics_data.append(metrics)
                
                if metrics_data:
                    df_metrics = pd.DataFrame(metrics_data)
                    
                    # Display metrics table
                    st.dataframe(
                        df_metrics[['Model', 'R²', 'RMSE', 'MAE', 'MAPE', 'Accuracy (%)']].style.format({
                            'R²': '{:.4f}',
                            'RMSE': '{:.2f}',
                            'MAE': '{:.2f}',
                            'MAPE': '{:.2f}%',
                            'Accuracy (%)': '{:.2f}%'
                        }).highlight_max(axis=0, subset=['R²', 'Accuracy (%)'], color='lightgreen')
                        .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen'),
                        use_container_width=True
                    )
                    
                    # Metrics visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_acc = go.Figure(data=[
                            go.Bar(
                                x=df_metrics['Model'],
                                y=df_metrics['Accuracy (%)'],
                                marker=dict(
                                    color=df_metrics['Accuracy (%)'],
                                    colorscale='Viridis',
                                    showscale=True
                                ),
                                text=df_metrics['Accuracy (%)'].round(2),
                                textposition='outside'
                            )
                        ])
                        fig_acc.update_layout(
                            title="Model Accuracy (%)",
                            yaxis_title="Accuracy (%)",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)
                    
                    with col2:
                        fig_r2 = go.Figure(data=[
                            go.Bar(
                                x=df_metrics['Model'],
                                y=df_metrics['R²'],
                                marker=dict(
                                    color=df_metrics['R²'],
                                    colorscale='Blues',
                                    showscale=True
                                ),
                                text=df_metrics['R²'].round(4),
                                textposition='outside'
                            )
                        ])
                        fig_r2.update_layout(
                            title="R² Score",
                            yaxis_title="R² Score",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_r2, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("📊 Prediction Statistics Summary")
                    
                    best_model = df_metrics.loc[df_metrics['R²'].idxmax(), 'Model']
                    best_r2 = df_metrics['R²'].max()
                    best_acc = df_metrics['Accuracy (%)'].max()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🏆 Best Model", best_model)
                    col2.metric("📈 Best R² Score", f"{best_r2:.4f}")
                    col3.metric("✅ Best Accuracy", f"{best_acc:.2f}%")
                    col4.metric("📊 Models Compared", len(df_metrics))
                    
                else:
                    st.error("❌ No valid predictions to compare")
    else:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Processing' tab")

# Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 **All Phases Complete**")
st.sidebar.markdown("✅ Phase 1: Data Engineering")
st.sidebar.markdown("✅ Phase 2: EDA & Baselines")
st.sidebar.markdown("✅ Phase 3: Advanced Models")
st.sidebar.markdown("✅ Phase 4: Validation")
st.sidebar.markdown("✅ Phase 5: Production Ready")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 **Models Included**")
st.sidebar.markdown("• Prophet (Facebook)")
st.sidebar.markdown("• SARIMA (Statistical)")
st.sidebar.markdown("• XGBoost (Gradient Boosting)")
st.sidebar.markdown("• LightGBM (Gradient Boosting)")
