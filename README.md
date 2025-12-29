# Stock Prediction ML Model

A machine learning application for predicting stock prices using LSTM (Long Short-Term Memory) neural networks. This project combines historical stock data analysis with deep learning to forecast future price trends.

## 🎯 Features

- **Real-time Stock Data**: Fetches current and historical stock data using yfinance API
- **LSTM Neural Network**: Advanced deep learning model for time series forecasting
- **Interactive Visualizations**: 
  - Heikin-Ashi candlestick charts with volume indicators
  - Moving average analysis (100-day and 200-day MA)
  - Predictions vs Original price comparison
- **Pre-trained Models**: Uses saved Keras models for fast predictions
- **User-friendly Interface**: Built with Streamlit for easy interaction

## 📊 Project Structure

```
Stock Prediction ML Model/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # Project license
├── src/
│   ├── __init__.py
│   └── Stockinfo.py                   # Main Streamlit application
├── models/
│   └── keras_model.h5                 # Pre-trained LSTM model
├── notebooks/
│   ├── LSTM model.ipynb               # Model training notebook
│   ├── LSTM model-checkpoint.ipynb    # Checkpoint notebook
│   ├── keras_model.h5                 # Saved model
│   └── my_model_keras.h5              # Alternative model
├── docs/
│   ├── architecture.md                # Detailed architecture documentation
│   └── README.md                      # Additional documentation
├── assets/                            # Project assets
└── price_pred_venv_fixed/             # Virtual environment (optional)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd "Stock Prediction ML Model"
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run src/Stockinfo.py
```

The app will open in your default browser at `http://localhost:8501`

## 📈 How It Works

### Data Collection
- Fetches historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
- Allows custom date ranges for analysis

### Data Processing
- **Normalization**: MinMax scaling (0-1 range) for stable neural network training
- **Heikin-Ashi Candles**: Smoothed candlestick representation for better trend visibility
- **Sequence Creation**: Converts time series into overlapping windows for LSTM training

### Model Architecture
```
Input Layer (100 timesteps × 1 feature)
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
LSTM Layer (60 units) + Dropout (0.3)
    ↓
LSTM Layer (80 units) + Dropout (0.4)
    ↓
LSTM Layer (120 units) + Dropout (0.5)
    ↓
Dense Layer (1 unit)
    ↓
Output (Predicted Price)
```

### Predictions
- Uses 70% of data for training, 30% for testing
- Predicts next day stock prices based on 100-day historical patterns
- Compares predictions against actual prices

## 💻 Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation and analysis |
| `matplotlib` | Static visualizations |
| `seaborn` | Statistical data visualization |
| `yfinance` | Fetch stock data from Yahoo Finance |
| `tensorflow` | Deep learning framework with Keras |
| `streamlit` | Interactive web app framework |
| `scikit-learn` | Data preprocessing (MinMaxScaler) |
| `altair` | Interactive charting library |

## 📝 Usage Guide

### Using the Application

1. **Enter Stock Ticker**: Input any valid stock symbol (e.g., AAPL, GOOGL, MSFT)
2. **Select Date Range**: Choose start and end dates for historical analysis
3. **View Analysis**:
   - Data statistics and descriptions
   - Closing price trends
   - Heikin-Ashi candlestick charts
   - Moving averages (100-day, 200-day)
   - Prediction results

### Example Tickers
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- TSLA (Tesla)
- AMZN (Amazon)

## 🔧 Training a New Model

To train a new LSTM model with different parameters:

1. Open `notebooks/LSTM model.ipynb`
2. Modify the model architecture or hyperparameters
3. Run the notebook cells
4. Save the trained model to `models/keras_model.h5`

## ⚠️ Important Notes

- **Internet Required**: The app needs an active internet connection to fetch stock data
- **Model Performance**: Past performance doesn't guarantee future results
- **Data Limitations**: 'Adj Close' may not be available for all tickers; the app uses 'Close' as fallback
- **Prediction Accuracy**: LSTM models work best with sufficient historical data (ideally 5+ years)

## 📊 Visualizations

### 1. Data Statistics
- Summary statistics of OHLCV data
- Data availability and quality checks

### 2. Heikin-Ashi Candlesticks
- **Green candles**: Bullish (closing > opening)
- **Red candles**: Bearish (closing < opening)
- Volume bars synchronized with price movements

### 3. Moving Averages
- 100-day moving average for short-term trends
- 200-day moving average for long-term trends
- Helps identify trend direction and support/resistance

### 4. Predictions
- Comparison graph of actual vs predicted prices
- Performance metrics visualization

## 🎓 Learning Resources

- **LSTM Networks**: [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Keras/TensorFlow**: [TensorFlow Documentation](https://www.tensorflow.org/)
- **Time Series Forecasting**: [Forecasting Principles](https://otexts.com/fpp2/)

## 🤝 Contributing

Feel free to fork, modify, and improve this project. Some ideas for enhancement:
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement ensemble models
- Add sentiment analysis from news/social media
- Create model comparison functionality
- Add portfolio prediction for multiple stocks

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🐛 Troubleshooting

### TensorFlow Import Error
```bash
pip install tensorflow --upgrade
```

### yfinance Data Issues
- Check internet connection
- Verify ticker symbol is valid
- Try a different date range

### Streamlit Not Found
```bash
pip install streamlit
```

### Model Loading Error
- Ensure `models/keras_model.h5` exists
- If missing, the app will train a new model automatically

## 📞 Support

For issues or questions:
1. Check the `docs/architecture.md` for technical details
2. Review the notebooks in the `notebooks/` directory
3. Examine the Streamlit app logs in the terminal

## ✨ Future Enhancements

- [ ] Multi-stock portfolio analysis
- [ ] Real-time predictions
- [ ] Risk assessment metrics
- [ ] Alternative models (GRU, Transformer, Prophet)
- [ ] Model performance metrics (RMSE, MAE)
- [ ] Data caching for faster reloads
- [ ] Advanced charting with technical indicators

---

**Last Updated**: December 2025
