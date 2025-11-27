# 🛂 Border Crossing Traffic Forecasting Platform

A comprehensive AI-powered forecasting system for US Border Crossing traffic prediction using multiple machine learning models with a premium analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### 📊 Advanced ML Models
- **Prophet** - Facebook's time series forecaster with holiday effects
- **SARIMA** - Statistical ARIMA model for seasonal patterns
- **XGBoost** - Gradient boosting with feature engineering
- **LightGBM** - High-performance gradient boosting
- **LSTM** - Deep learning for complex patterns
- **Ensemble Methods** - Stacking and weighted averaging

### 🎯 Key Capabilities
- ✅ **Automated Data Pipeline** - Clean, validate, and process data
- ✅ **Feature Engineering** - Lag features, rolling statistics, cyclical encoding
- ✅ **Hyperparameter Tuning** - Optuna-based optimization
- ✅ **Rolling Cross-Validation** - Time series-aware validation
- ✅ **Model Comparison** - Train on ≤2024, test on 2025
- ✅ **Premium UI** - Beautiful analytics-themed dashboard
- ✅ **Comprehensive Metrics** - R², RMSE, MAE, MAPE, Accuracy

### 🎨 Premium Dashboard
- Modern glassmorphism design
- Analytics-themed background with gradients
- Interactive visualizations with Plotly
- Real-time model training and comparison
- Automated best model selection

## 📁 Project Structure

```
BTS/
├── data/
│   ├── raw/              # Raw data files (gitignored)
│   └── processed/        # Cleaned data (gitignored)
├── src/
│   ├── data/             # Data loading and cleaning
│   │   ├── loader.py
│   │   ├── cleaning.py
│   │   └── fetcher.py
│   ├── features/         # Feature engineering
│   │   └── engineering.py
│   ├── models/           # Model implementations
│   │   ├── baseline.py   # Prophet
│   │   ├── sarima.py     # SARIMA
│   │   ├── tree_models.py # XGBoost, LightGBM
│   │   ├── deep_learning.py # LSTM
│   │   ├── ensemble.py   # Ensemble methods
│   │   └── tuning.py     # Hyperparameter tuning
│   ├── evaluation/       # Metrics and validation
│   │   ├── metrics.py
│   │   └── cross_validation.py
│   ├── dashboard.py      # Streamlit web app
│   ├── main.py           # CLI pipeline
│   └── styles.py         # Custom CSS
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/border-crossing-forecasting.git
cd border-crossing-forecasting
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run src/dashboard.py
```

Or use the PowerShell script:
```powershell
.\run_pipeline.ps1
```

## 📊 Usage

### 1. Upload Data
- Go to the "Data Upload & Processing" tab
- Upload your CSV/Excel file with border crossing data
- The system will automatically clean and process the data

### 2. Run Model Comparison
- Navigate to the "Complete Model Comparison & Analytics" tab
- Click "Train All Models & Generate Comparison"
- Wait for all models to train (may take a few minutes)

### 3. View Results
- **Predictions vs Actual** - Interactive chart showing all model predictions
- **Performance Metrics** - R², RMSE, MAE, MAPE, Accuracy for each model
- **Best Model** - Automatically identified based on R² score
- **Statistics Summary** - Complete overview of model performance

## 📈 Model Performance

The system trains models on data up to December 2024 and tests on 2025 data for realistic performance assessment.

**Target Metrics:**
- R² ≥ 0.92 (through ensemble methods)
- MAPE < 10% (for stable, high-volume ports)
- Robust predictions with confidence intervals

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### CLI Pipeline
```bash
python src/main.py
```

## 📚 Data Sources

The system works with Border Crossing Entry Data from the Bureau of Transportation Statistics (BTS).

**Required Columns:**
- `date` - Timestamp of the crossing
- `value` - Number of crossings
- `port_name` - Name of the border port (optional)
- `measure` - Type of crossing (pedestrian/vehicle/commercial) (optional)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Bureau of Transportation Statistics for the data
- Facebook Prophet team
- XGBoost and LightGBM developers
- Streamlit team for the amazing framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Python, Streamlit, and advanced ML techniques**
