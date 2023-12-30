Certainly! Below is a sample README file for your GitHub repository:

---

# Portfolio Optimization using Deep Learning

## Overview

This Python script demonstrates portfolio optimization using both traditional optimization techniques and deep learning. The project is inspired by Zhiao Zhang's paper "Deep Learning for Portfolio Optimization."

## Description

The script fetches historical data for a selected set of ETFs, calculates daily returns, and applies both traditional optimization and deep learning to determine optimal portfolio weights. The chosen optimization methods aim to maximize the Sharpe Ratio, a metric that considers both returns and risk.

## Requirements

- Python 3
- Required Python packages: pandas, yfinance, numpy, tensorflow, sklearn, scipy, matplotlib

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/simonlesflex/portfolio-optimization.git
   ```

2. Install the required packages:

   ```bash
   pip install pandas yfinance numpy tensorflow sklearn scipy matplotlib
   ```

3. Execute the script:

   ```bash
   python mainPortfolioDeepOptimizer_p.py --start_date 2016-01-01 --end_date 2023-12-31
   ```

## ETF Symbols

Modify the `etf_symbols` list in the script to choose the assets for portfolio optimization. Examples include:

- US Market: ['SPY', 'QQQ', 'TLT', 'QLD', 'PSQ', 'SHV', 'IEF', 'SSO', 'QID', 'SMH', 'USD', 'GLD', 'UUP', 'IEI']
- European Market: ['CSSPX.MI', 'EQQQ.DE', 'QQQ3.MI', 'XS2D.L', 'LVE.PA', 'AEEM.PA', 'SGLD.MI', 'XGSH.MI', 'CSBGU7.MI', 'ZPRV.DE', 'SXLK.MI', 'XDWH.DE']

## Results

The script displays the optimized weights for both traditional optimization and deep learning prediction. Additionally, a pie plot is generated to visualize the asset allocation in each case.

## Acknowledgments

- Zhiao Zhang's paper: [Deep Learning for Portfolio Optimization](https://arxiv.org/abs/1912.00534)

---

Feel free to adjust the sections and content based on your specific needs and additional details about the project.
