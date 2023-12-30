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
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta

# Define the ETF symbols
#etf_symbols = ['SPY', 'QQQ', 'BND', 'AGG', 'GLD', 'AGGH', 'IWM', 'VTV', 'VUG', 'XLK']
#etf_symbols = ['CSSPX.MI', 'EQQQ.DE', 'QQQ3.MI', 'XS2D.L', 'LVE.PA', 'AEEM.PA', 'SGLD.MI', 'XGSH.MI', 'CSBGU7.MI', 'ZPRV.DE', 'SXLK.MI', 'XDWH.DE']
#etf_symbols = ['CSSPX.MI', 'EQQQ.DE', 'QQQ3.MI', 'QDVI.DE', 'XS2D.L', '3FNE.L', '0W9J.IL', 'LVE.PA', 'EMVL.L', 'SGLD.MI', 'XGSH.MI', 'CSBGU7.MI', 'ZPRV.DE', 'SXLK.MI', 'XDWH.DE']
#etf_symbols = ['VTI', 'AGG', 'DBC', 'VIXY']
etf_symbols = ['SPY', 'QQQ', 'TLT', 'QLD', 'SHV', 'IEF', 'SSO', 'SMH', 'USD', 'GLD', 'UUP', 'IEI']

# Set warmup period
n_periods = 50
# Initial Capital
G_INITCAP = 10000

def parse_arguments():
    parser = argparse.ArgumentParser(description='Portfolio Optimization using Deep Learning')
    parser.add_argument('--start_date', type=str, help='Start date in the format YYYY-MM-DD', required=False)
    parser.add_argument('--end_date', type=str, help='End date in the format YYYY-MM-DD', required=False)
    args = parser.parse_args()

    # Set default values if arguments are empty
    if not args.start_date:
        args.start_date = ''  # Set your default start date
    if not args.end_date:
        args.end_date = ''  # Set your default end date

    return args

def download_data(etf_symbols, warm_date, start_date, end_date):
    dataall = yf.download(etf_symbols, start=warm_date, end=end_date)['Adj Close']
    returnsall = dataall.pct_change().dropna()
    retnormall_df = returnsall
    returnssub = returnsall[start_date:]
    data = dataall
    
    scaler = MinMaxScaler()
    returns_normalized = scaler.fit_transform(retnormall_df)
    return returns_normalized, retnormall_df, returnssub, data

def performance_metrics(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    cumulative_returns = np.cumprod(1 + returns) - 1
    drawdown = np.min(1 - cumulative_returns)  # Corrected drawdown calculation
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
    
    cagr = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1  # CAGR calculation

    
    print(f"Daily Returns: {returns[-1]*100:.2f}%")
    print(f"Cumulative Returns: {cumulative_returns[-1]*100:.2f}%")
    print(f"Max Drawdown: {drawdown*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"CAGR: {cagr*100:.2f}%")

def main():
    args = parse_arguments()
    # Set default values if arguments are empty
    if len(args.start_date) == 0: 
        start_date = datetime(2018, 1, 1)
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    if len(args.end_date) == 0:
        end_date = datetime.today()
    else:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    warmdate = start_date - timedelta(n_periods*1.5)
    # Download historical data
    retnormall, retnormall_df, histret, histalldata = download_data(etf_symbols, warmdate, start_date, end_date)
    retnormall_df = retnormall_df.shift()
    
    lendif = len(retnormall_df) - len(histret)
    
    # Initialize variables
    data_window = []
    model = None
    portfolio_values = []
    MonthlyPortfolioW = []
    month = 0
    
    # Iterate on a daily basis
    for index, row in histret.iterrows():
    
        # Use a rolling window of historical data for each iteration
        window_data = retnormall_df.loc[:(index)][-n_periods:]
        
        if month != 0:
            # Measure daily performance
            daily_returns = np.sum(MonthlyPortfolioW * retnormall_df.loc[index])
            if not portfolio_values:
                portfolio_values.append(G_INITCAP * (1 + daily_returns))
            else:
                portfolio_values.append(portfolio_values[-1] * (1 + daily_returns))
    
        # Check if a month has passed to trigger rebalance
        if month != index.month:
            month = index.month
            datestr = str(index)[:10]
            # Split data into features (X) and target (y)
            train_size = int(len(window_data) * 0.5)
            X_train, X_test = window_data.iloc[:train_size], window_data.iloc[train_size:]
            
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(len(etf_symbols),)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(len(etf_symbols), activation='softmax')
            ])
            
            model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001), loss='mse')
            
            # Train the model
            model.fit(X_train, X_test, epochs=10, batch_size=1, verbose=0)
    
    
            # Predict optimized weights
            predicted_weights = model.predict(np.array([window_data.iloc[-1]]))[0]
    
            # Display results
            print(f"Predicted Weights for Day {datestr}: {predicted_weights}")
    
        MonthlyPortfolioW = np.round(predicted_weights, 2)
        
    
    # Display performance metrics
    performance_metrics(portfolio_values)
    
    # Plot portfolio values
    dates = histret.index[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, portfolio_values, label='Portfolio Value', color='blue')
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()
    
    # Pie plot
    labels = etf_symbols
    plt.figure(figsize=(12, 6))
    
    # Deep Learning Prediction for the last window
    plt.pie(predicted_weights, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Deep Learning Prediction Portfolio Weights')
    
    plt.show()




if __name__ == "__main__":
    main()


