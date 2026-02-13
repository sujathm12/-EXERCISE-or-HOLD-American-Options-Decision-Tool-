---
title: American Option Early Exercise Visualizer
emoji: 📈
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# American Option Early Exercise Visualizer

Based on Steven Shreve's "Stochastic Calculus for Finance I" - Chapter 4

## Features

- **Binomial Tree Pricing**: Implements Cox-Ross-Rubinstein method
- **Early Exercise Boundary**: Visualizes exactly when to exercise
- **Shreve Proof**: Interactive demonstration of why calls shouldn't be exercised early
- **Real-time Data**: Fetches current stock prices via yfinance
- **Exercise Decision**: Clear "Exercise vs Hold" recommendation

## How to Use

1. Enter stock ticker or manual price
2. Set option parameters (strike, type)
3. Adjust market parameters (volatility, interest rate)
4. Click "Calculate"
5. View the exercise boundary and decision

## Mathematical Foundation

The binomial tree implements backward induction with early exercise checks:
V_n(s) = max{g(s), (p̃V_{n+1}(us) + q̃V_{n+1}(ds))/(1+r)}

## Deployment

This app is deployed on Hugging Face Spaces and runs entirely in the browser.
