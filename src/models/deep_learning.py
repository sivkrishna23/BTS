import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_lstm(df: pd.DataFrame, target_col: str = 'value', 
               seq_length: int = 60, epochs: int = 50, 
               batch_size: int = 64) -> Dict[str, Any]:
    """
    Train LSTM model for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        seq_length: Sequence length for LSTM
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Dictionary with model, scaler, and predictions
    """
    # Prepare data
    data = df[target_col].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length)
    
    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Reshape for LSTM (samples, seq_length, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=2, dropout=0.2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    logger.info("Training LSTM model...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    train_preds, test_preds = [], []
    
    with torch.no_grad():
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            train_preds.extend(preds.cpu().numpy())
        
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            test_preds.extend(preds.cpu().numpy())
    
    # Inverse transform predictions
    train_preds = scaler.inverse_transform(np.array(train_preds).reshape(-1, 1)).flatten()
    test_preds = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
    
    y_train_actual = scaler.inverse_transform(y_train).flatten()
    y_test_actual = scaler.inverse_transform(y_test).flatten()
    
    return {
        'model': model,
        'scaler': scaler,
        'train_pred': train_preds,
        'test_pred': test_preds,
        'y_train': y_train_actual,
        'y_test': y_test_actual,
        'seq_length': seq_length
    }
