# PCA-Based Statistical Arbitrage (Pair Trading)

## Project Overview
This project implements a **Statistical Arbitrage** strategy using **Principal Component Analysis (PCA)** to identify and trade mean-reverting pairs in a simulated stock market environment.

The goal is to demonstrate advanced quantitative finance techniques, including dimensionality reduction for factor extraction and cointegration testing for pair selection.

## Methodology

### 1. Data Simulation
We simulate a universe of **50 stocks** driven by **3 latent market factors** (e.g., Market, Sector A, Sector B). This ensures that the underlying mathematical relationships exist, allowing for a clean demonstration of the strategy logic without the noise and cost of real-time market data APIs.

### 2. Factor Extraction (PCA)
We apply **Principal Component Analysis (PCA)** to the returns matrix to extract the top $K$ components that explain the majority of the market variance.
- **Systematic Risk**: The part of returns explained by these PCA components.
- **Idiosyncratic Risk (Residuals)**: The remaining part of returns (`Actual - Systematic`).

### 3. Pair Selection
We analyze the **residuals** of all stocks to find pairs that are highly correlated *after* removing market risk.
- A high correlation in residuals implies that two stocks behave similarly for reasons specific to them, making them ideal candidates for pair trading.

### 4. Trading Strategy
We construct a spread between the selected pair and trade based on **Z-Score Mean Reversion**:
- **Long the Spread**: When Z-Score < -2.0 (Spread is statistically too low).
- **Short the Spread**: When Z-Score > 2.0 (Spread is statistically too high).
- **Exit**: When Z-Score returns to 0.

## Files
- `pca_pair_trading.ipynb`: The main Jupyter Notebook containing all code, visualizations, and backtest results.

## How to Run

### Option A: Jupyter Notebook (Local)
1.  Ensure you have Python installed with `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.
2.  Open the notebook:
    ```bash
    jupyter notebook pca_pair_trading.ipynb
    ```

### Option B: Docker (Reproducible Environment)
This project includes a Dockerfile to run the Gradient Descent simulation in a consistent environment.

1.  **Build the Image:**
    ```bash
    docker build -t pca-trading .
    ```
2.  **Run the Container:**
    ```bash
    docker run pca-trading
    ```
    *(Note: This runs the `gradient_descent_scratch.py` script as a demo).*
