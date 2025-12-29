# Architecture of Stock-Prediction-ML-Model

This document outlines the architecture of the `Stock-Prediction-ML-Model`, a machine learning model designed to predict stock prices using Long Short-Term Memory (LSTM) networks implemented with Keras. The model is structured to process historical stock data and generate future price predictions.

## Overview
The model leverages LSTM, a type of recurrent neural network (RNN) suited for time series forecasting, to capture long-term dependencies in stock price data. It is designed to handle sequential data, making it ideal for predicting future stock movements based on past trends.

## Data Flow
1. **Data Input**:
   - Historical stock data (e.g., opening price, closing price, volume) is sourced from external datasets or APIs.
   - Data is stored in the `data/` directory (e.g., `data/raw/` for raw files, `data/processed/` for preprocessed data).

2. **Preprocessing**:
   - Data is cleaned and normalized (e.g., scaling between 0 and 1) in the `src/stockinfo.py` script.
   - Features are extracted and sequenced into time steps for LSTM input.

3. **Model Training**:
   - The `notebooks/lstm_model.ipynb` contains the training pipeline, including data loading, model definition, and evaluation.
   - The trained model is saved as `models/keras_model.h5`.

4. **Prediction**:
   - The model generates predictions for future stock prices, which can be visualized or analyzed in the notebook.

## Model Architecture
The LSTM model consists of the following layers:
- **Input Layer**: Accepts a 3D tensor of shape `(samples, time_steps, features)` (e.g., 50 time steps, 5 features like open, close, high, low, volume).
- **LSTM Layers**: 
  - A stack of LSTM layers (e.g., 50 units each) to learn temporal patterns.
  - Dropout layers (e.g., 0.2) to prevent overfitting.
- **Dense Layer**: A fully connected layer to output the predicted price.
- **Output Layer**: A single neuron with a linear activation for regression (predicting continuous price values).

### Example Configuration
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')