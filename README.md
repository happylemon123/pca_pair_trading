# PCA-Based Statistical Arbitrage & Regime Detection

## Project Overview
This repository implements a **Statistical Arbitrage** strategy that leverages **Principal Component Analysis (PCA)** for factor modeling and **Hidden Markov Models (HMM)** for market regime detection. The system identifies identifying cointegrated pairs by isolating idiosyncratic risk (residuals) after hedging systematic market factors.

The core objective is to demonstrate a robust quantitative trading framework that adapts to changing market conditions, specifically filtering trades based on latent volatility states.

## Key Components

### 1. Factor Model (PCA)
The strategy decomposes asset returns into:
- **Systematic Risk**: Captures market-wide movements using the top $K$ principal components.
- **Idiosyncratic Risk (Residuals)**: Isolates stock-specific alpha (`Actual - Systematic`).

PCA is applied to a universe of 50 assets to extract latent factors, ensuring that the resulting signals are orthogonal to broad market movements.

### 2. Signal Generation (Mean Reversion)
We model the spreads of selected pairs as an **Ornstein-Uhlenbeck** process. Trading signals are generated based on Z-Score thresholds:
- **Entry**: When the spread deviates significantly from its historical mean (e.g., Z-Score > 2.0 or < -2.0).
- **Exit**: When the spread reverts to the mean (Z-Score approaching 0).

### 3. Risk Management (HMM Regime Switching)
A **Hidden Markov Model (HMM)** is implemented to classify market states into distinct regimes:
- **Calm / Bullish**: Low volatility, high correlation stability. Suitable for mean-reversion strategies.
- **Volatile / Bearish**: High volatility, breakdown of historical correlations. Trading is paused to mitigate tail risk.

This acts as a dynamic filter, preventing the strategy from executing trades during periods where statistical assumptions are likely to fail.

## Architecture

- **`pca_pair_trading.ipynb`**: The primary research environment containing the full backtesting pipeline. Includes data generation, factor analysis, signal construction, and performance attribution.
- **`market_regime_hmm.py`**: A standalone module for the HMM implementation. Demonstrates the training of the regime detection model using Gaussian Mixture inputs.
- **`gradient_descent_scratch.py`**: Simulation engine for generating synthetic price paths with controlled drift and volatility parameters, used to validate the strategy's theoretical underpinnings.

## Usage

### Simulated Environment
The strategy uses a simulated data feed to ensure reproducibility and to isolate the performance of the logic from external market noise. The simulation generates 3 latent factors driving 50 stocks, providing a clean ground truth for testing cointegration.

### Running with Docker
A Dockerfile is provided to ensure a consistent execution environment.

1.  **Build the Image:**
    ```bash
    docker build -t pca-trading .
    ```

2.  **Run the Container:**
    ```bash
    docker run pca-trading
    ```

### Running Locally
Ensure Python 3.8+ is installed with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

```bash
pip install -r requirements.txt
jupyter notebook pca_pair_trading.ipynb
```
