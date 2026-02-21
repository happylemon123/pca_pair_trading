# Thinking Process: From Static PCA to Dynamic Kalman Filter in Pairs Trading

This document summarizes the chronological thought process and mathematical evolution of our pairs trading strategy, specifically transitioning from a basic PCA approach to an advanced, dynamically calibrated Kalman Filter.

## 1. The Flaw of the "Dumb" Z-Score
The standard textbook approach to statistical arbitrage (and our initial PCA approach) relies on a simple rolling Z-Score (e.g., a 60-day moving average and standard deviation).
*   **The Assumption:** Market relationships are static over that 60-day window. It assumes the "hedge ratio" (the optimal proportion of Stock A vs Stock B) remains constant.
*   **The Reality:** Markets are non-stationary. A relationship that existed 60 days ago might have fundamentally shifted yesterday. Furthermore, highly cointegrated stocks (like CDNS and SNPS) move so closely that a static model waiting for a perfect $2\sigma$ deviation will often result in zero trades.

## 2. The Solution: The Kalman Filter
To make the strategy dynamic, we replaced the static rolling window with a **Kalman Filter**.
*   **What it does:** A Kalman Filter is an algorithm that estimates a "hidden state" (the true, unobservable hedge ratio) based on a series of "noisy observations" (the daily stock prices).
*   **Dynamic Hedge Ratio:** Instead of a flat line, the Kalman Filter updates the hedge ratio every single day based on incoming price data, adapting instantly to market shifts.
*   **Dynamic Z-Score:** At each step, the Kalman Filter mathematically calculates the variance (uncertainty) of its own prediction. We use this live variance to calculate the Z-Score, meaning the $1.5\sigma$ threshold naturally expands and contracts based on local market behavior.

## 3. The Calibration Problem: Maximum Likelihood & EM Algorithm
A Kalman Filter requires two critical hyper-parameters:
*   **$Q$ (Transition Covariance):** The variance of the hidden state. How fast is the fundamental relationship actually changing?
*   **$R$ (Observation Covariance):** The variance of the observational noise. How "noisy" or "bouncy" is the daily stock market?

If you guess these numbers arbitrarily, the filter is useless. We must calibrate them using rigorous statistics.
*   **Method of Moments (MoM):** This involves calculating the sample mean and variance of the historical data to estimate parameters. It is computationally cheap but often inaccurate for complex latent variable models.
*   **Maximum Likelihood Estimate (MLE):** This is the gold standard. We want the parameters that make our historical data the "most likely" to have occurred. 
*   **Expectation-Maximization (EM):** Because our state (the true hedge ratio) is hidden, we cannot calculate the MLE directly. We use the EM algorithm. It alternates between estimating the hidden state (Expectation) and optimizing $Q$ and $R$ to maximize the likelihood (Maximization). 
*   **Our Implementation:** We split our CDNS/SNPS data (Year 1 is In-Sample, Years 2-3 are Out-of-Sample). We ran `.em()` on Year 1, strictly locking in $Q$ at exactly $0.000019$ and $R$ at $0.567$.

## 4. The Final Form: Surviving 6-Sigma Events (Fat Tails)
A common critique of Z-Score models is their failure to account for "Fat Tails." If returns were perfectly normally distributed, a $6\sigma$ event should happen once every 1.38 million years. In reality, market crashes happen every decade. A static model blows up during these events.

Our dynamically calibrated Kalman Filter survived the September 2025 tech-sector rotation anomaly because it dynamically recognized the variance expansion. 

### What's Next? The Ultimate Quant Combo
To further insulate against Fat Tails, advanced quant funds combine techniques:
1.  **GARCH Volatility Injection:** Our basic Kalman Filter assumes observation noise ($R$) is constant. GARCH predicts *tomorrow's* specific volatility based on *today's*. By predicting the daily GARCH variance and feeding it directly into the Kalman Filter as a time-varying $R_t$ parameter, the filter becomes hyper-aware of volatility clustering (market crashes).
2.  **Hidden Markov Regime Switching (HMM):** Train an HMM to detect distinct market regimes (e.g., "Normal" vs "High-Volatility Crisis"). If the HMM detects a Crisis regime, the strategy is mathematically forced to halt trading, completely neutralizing the risk of a $6\sigma$ blowout.
