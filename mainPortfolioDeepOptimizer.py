#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:49:07 2023
https://europoor.com/how-to-buy-leveraged-etfs-from-europe/
@author: simonlesflex
"""

import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the ETF symbols
#etf_symbols = ['SPY', 'QQQ', 'BND', 'AGG', 'GLD', 'AGGH', 'IWM', 'VTV', 'VUG', 'XLK']
#etf_symbols = ['CSSPX.MI', 'EQQQ.DE', 'QQQ3.MI', 'XS2D.L', 'LVE.PA', 'AEEM.PA', 'SGLD.MI', 'XGSH.MI', 'CSBGU7.MI', 'ZPRV.DE', 'SXLK.MI', 'XDWH.DE']
#etf_symbols = ['CSSPX.MI', 'EQQQ.DE', 'QQQ3.MI', 'XS2D.L', '3FNE.L', '0W9J.IL', 'LVE.PA', 'EMVL.L', 'SGLD.MI', 'XGSH.MI', 'CSBGU7.MI', 'ZPRV.DE', 'SXLK.MI', 'XDWH.DE']
#etf_symbols = ['VTI', 'AGG', 'DBC', 'VIXY']
etf_symbols = ['SPY', 'QQQ', 'TLT', 'QLD', 'PSQ', 'SHV', 'IEF', 'SSO', 'QID', 'SMH', 'USD', 'GLD', 'UUP', 'IEI']
# Download historical data
data = yf.download(etf_symbols, start='2022-01-01', end='2023-01-01')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Function to calculate Sharpe Ratio
def sharpe_ratio(weights, returns):
    portfolio_return = np.dot(returns.mean(), weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    return -portfolio_return / portfolio_volatility

# Normalizing data
scaler = MinMaxScaler()
returns_normalized = scaler.fit_transform(returns)

# Split data into features (X) and target (y)
X = returns_normalized[:-1]
y = returns_normalized[1:]

# Build a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(len(etf_symbols),)),
    keras.layers.Dense(len(etf_symbols), activation='softmax')
])

model.compile(optimizer='adam', loss='mse')  # Using mean squared error as a loss function

# Train the model
model.fit(X, y, epochs=50, batch_size=1, verbose=0)

# Predict optimized weights
predicted_weights = model.predict(np.array([returns_normalized[-1]]))[0]

# Constraint: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})

# Bounds: weights between 0 and 1
bounds = tuple((0, 1) for _ in range(len(etf_symbols)))

# Optimization
result = minimize(sharpe_ratio, np.ones(len(etf_symbols)) / len(etf_symbols),
                  args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimized weights from traditional optimization
traditional_weights = result.x

# Display results
print("Traditional Optimization Weights:", traditional_weights)
print("Deep Learning Predicted Weights:", predicted_weights)

# Create a DataFrame
predicted_portfolio_df = pd.DataFrame(list(zip(etf_symbols, predicted_weights)), columns=['Asset Ticker', 'Weight'])

# Display the DataFrame
print(predicted_portfolio_df)


# Pie plot
labels = etf_symbols
plt.figure(figsize=(12, 6))

# Traditional Optimization
plt.subplot(1, 2, 1)
plt.pie(traditional_weights, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Traditional Optimization Portfolio Weights')

# Deep Learning Prediction
plt.subplot(1, 2, 2)
plt.pie(predicted_weights, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Deep Learning Prediction Portfolio Weights')

plt.show()