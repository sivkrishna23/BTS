"""
Comprehensive Exploratory Data Analysis Module
Advanced visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

def create_distribution_analysis(df, value_col='value'):
    """Create distribution analysis visualizations."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution Histogram', 'Box Plot', 'Q-Q Plot', 'Violin Plot'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'violin'}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[value_col], name='Distribution', marker_color='#667eea', nbinsx=50),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=df[value_col], name='Box Plot', marker_color='#764ba2'),
        row=1, col=2
    )
    
    # Q-Q plot
    qq = stats.probplot(df[value_col].dropna(), dist="norm")
    fig.add_trace(
        go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Q-Q Plot', marker=dict(color='#f093fb')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=qq[0][0], y=qq[1][1] + qq[1][0]*qq[0][0], mode='lines', name='Theoretical', line=dict(color='red')),
        row=2, col=1
    )
    
    # Violin plot
    fig.add_trace(
        go.Violin(y=df[value_col], name='Violin Plot', fillcolor='#4facfe', line_color='#00f2fe'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Distribution Analysis",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_time_series_decomposition(df, value_col='value', timestamp_col='timestamp'):
    """Create seasonal decomposition visualization."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Prepare data
    df_ts = df.set_index(timestamp_col)[value_col].asfreq('D')
    df_ts = df_ts.fillna(method='ffill')
    
    # Decompose
    decomposition = seasonal_decompose(df_ts, model='additive', period=365)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.08
    )
    
    # Original
    fig.add_trace(
        go.Scatter(x=df_ts.index, y=df_ts.values, name='Original', line=dict(color='#667eea')),
        row=1, col=1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend', line=dict(color='#764ba2')),
        row=2, col=1
    )
    
    # Seasonal
    fig.add_trace(
        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal', line=dict(color='#f093fb')),
        row=3, col=1
    )
    
    # Residual
    fig.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual', line=dict(color='#4facfe')),
        row=4, col=1
    )
    
    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="Time Series Decomposition",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_temporal_patterns(df, value_col='value', timestamp_col='timestamp'):
    """Create temporal pattern visualizations."""
    df = df.copy()
    df['hour'] = pd.to_datetime(df[timestamp_col]).dt.hour
    df['day_of_week'] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
    df['month'] = pd.to_datetime(df[timestamp_col]).dt.month
    df['year'] = pd.to_datetime(df[timestamp_col]).dt.year
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('By Day of Week', 'By Month', 'By Year', 'By Hour'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # By day of week
    dow_data = df.groupby('day_of_week')[value_col].mean()
    fig.add_trace(
        go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], y=dow_data.values, 
               marker_color='#667eea', name='Day of Week'),
        row=1, col=1
    )
    
    # By month
    month_data = df.groupby('month')[value_col].mean()
    fig.add_trace(
        go.Bar(x=month_data.index, y=month_data.values, marker_color='#764ba2', name='Month'),
        row=1, col=2
    )
    
    # By year
    year_data = df.groupby('year')[value_col].mean()
    fig.add_trace(
        go.Bar(x=year_data.index, y=year_data.values, marker_color='#f093fb', name='Year'),
        row=2, col=1
    )
    
    # By hour (if hour data exists)
    if df['hour'].nunique() > 1:
        hour_data = df.groupby('hour')[value_col].mean()
        fig.add_trace(
            go.Bar(x=hour_data.index, y=hour_data.values, marker_color='#4facfe', name='Hour'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Temporal Patterns Analysis",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_statistical_summary(df, value_col='value'):
    """Create comprehensive statistical summary."""
    stats_dict = {
        'Count': len(df),
        'Mean': df[value_col].mean(),
        'Median': df[value_col].median(),
        'Std Dev': df[value_col].std(),
        'Min': df[value_col].min(),
        'Max': df[value_col].max(),
        'Q1 (25%)': df[value_col].quantile(0.25),
        'Q3 (75%)': df[value_col].quantile(0.75),
        'IQR': df[value_col].quantile(0.75) - df[value_col].quantile(0.25),
        'Skewness': df[value_col].skew(),
        'Kurtosis': df[value_col].kurtosis(),
        'CV (%)': (df[value_col].std() / df[value_col].mean()) * 100
    }
    
    return pd.DataFrame(list(stats_dict.items()), columns=['Statistic', 'Value'])
