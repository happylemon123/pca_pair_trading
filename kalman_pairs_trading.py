import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import os

def load_data(filepath, stock_y='CDNS', stock_x='SNPS'):
    """Loads stock data and returns the two target series."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Please ensure data exists.")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    if stock_y not in df.columns or stock_x not in df.columns:
        raise ValueError(f"Stocks {stock_y} and/or {stock_x} not found in data.")
        
    return df[[stock_y, stock_x]].dropna()

def calibrate_kalman(train_x, train_y):
    """
    Use Expectation-Maximization to calibrate the Transition and Observation 
    Covariances for the Kalman Filter based on In-Sample (training) data.
    
    Model:
    y_t = x_t * beta_t + v_t  (Observation Equation)
    beta_t = beta_{t-1} + w_t (State Equation)
    """
    print(f"Calibrating Kalman Filter using {len(train_x)} training samples...")
    
    # Formulate the time-varying observation matrix
    # shape: (n_timesteps, observation_dim, state_dim) = (N, 1, 1)
    obs_mat = train_x.values.reshape(-1, 1, 1)
    
    # Initialize Kalman Filter
    # We set initial guesses, and tell EM to optimize the covariances.
    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=1,
        initial_state_mean=np.ones(1),
        initial_state_covariance=np.ones((1, 1)),
        transition_matrices=np.ones((1, 1)),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=0.01
    )
    
    # Run EM to find Maximum Likelihood Estimates for process & observation noise
    kf = kf.em(train_y.values, n_iter=10, em_vars=['transition_covariance', 'observation_covariance'])
    
    print(f"Calibrated Transition Covariance (Q): {kf.transition_covariance[0,0]:.6f}")
    print(f"Calibrated Observation Covariance (R): {kf.observation_covariance[0,0]:.6f}")
    
    return kf

def apply_dynamic_strategy(kf, df_x, df_y, threshold=1.5):
    """
    Run the calibrated Kalman filter over the dataset to get dynamic hedge ratios
    and execute the trading strategy based on dynamic z-scores.
    """
    print(f"Running dynamic filter over {len(df_x)} total samples...")
    
    obs_mat = df_x.values.reshape(-1, 1, 1)
    
    # We update the kf observation matrices for the full dataset before filtering
    kf.observation_matrices = obs_mat
    
    # Filter the data to get the hidden state (dynamic beta / hedge ratio) and state covariance
    state_means, state_covs = kf.filter(df_y.values)
    
    hedge_ratios = state_means.flatten()
    
    # Reconstruct the expected price of Y
    expected_y = hedge_ratios * df_x.values
    
    # Calculate the spread (prediction error / innovation)
    spread = df_y.values - expected_y
    
    # Calculate the variance of the prediction error
    # Variance(y_t) = x_t * Cov(beta_t) * x_t^T + R
    pred_error_var = (df_x.values ** 2) * state_covs.flatten() + kf.observation_covariance[0, 0]
    pred_error_std = np.sqrt(pred_error_var)
    
    # Dynamic Z-Score
    z_score = spread / pred_error_std
    
    # Create output dataframe
    results = pd.DataFrame(index=df_y.index)
    results['Y'] = df_y
    results['X'] = df_x
    results['Hedge_Ratio'] = hedge_ratios
    results['Spread'] = spread
    results['Z_Score'] = z_score
    
    # Trading Logic
    # Enter when Z-score crosses threshold, exit when it crosses 0
    positions = np.zeros(len(results))
    current_pos = 0
    
    for i in range(len(results)):
        z = z_score[i]
        
        # Entry logic
        if z > threshold and current_pos == 0:
            current_pos = -1  # Short Spread: Short Y, Long X
        elif z < -threshold and current_pos == 0:
            current_pos = 1   # Long Spread: Long Y, Short X
            
        # Exit logic
        elif current_pos == -1 and z <= 0:
            current_pos = 0
        elif current_pos == 1 and z >= 0:
            current_pos = 0
            
        positions[i] = current_pos
        
    results['Position'] = positions
    
    # Shifts positions to calculate PnL so we don't look ahead
    # (trade is executed at the close, effects next day's return)
    results['Position_Y'] = results['Position'].shift(1).fillna(0)
    results['Position_X'] = -results['Position'].shift(1).fillna(0) * results['Hedge_Ratio'].shift(1).fillna(0)
    
    # Calculate daily returns for individual stocks
    pct_returns_y = results['Y'].pct_change().fillna(0)
    pct_returns_x = results['X'].pct_change().fillna(0)
    
    # Portfolio daily returns
    results['Daily_PnL'] = (results['Position_Y'] * pct_returns_y) + (results['Position_X'] * pct_returns_x)
    results['Cumulative_PnL'] = results['Daily_PnL'].cumsum()
    
    return results

def plot_results(results):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Plot 1: Dynamic Hedge Ratio
    axes[0].plot(results.index, results['Hedge_Ratio'], color='purple', label='Dynamic Hedge Ratio (\u03b2)')
    axes[0].set_title('Kalman Filter: Dynamic Hedge Ratio Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Dynamic Z-Score and Thresholds
    axes[1].plot(results.index, results['Z_Score'], color='royalblue', label='Dynamic Z-Score')
    axes[1].axhline(1.5, color='red', linestyle='--', label='Short Threshold (+1.5)')
    axes[1].axhline(-1.5, color='green', linestyle='--', label='Long Threshold (-1.5)')
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_title('Dynamic Spread Z-Score (Note: adapts dynamically to localized volatility)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative PnL
    axes[2].plot(results.index, results['Cumulative_PnL'], color='green', label='Strategy PnL')
    axes[2].set_title('Cumulative Equity Curve')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kalman_pairs_results.png')
    print("Saved plot to kalman_pairs_results.png")
    
if __name__ == "__main__":
    data_path = 'data/stock_prices.csv'
    
    # 1. Load Data
    print("Loading data...")
    df = load_data(data_path, stock_y='CDNS', stock_x='SNPS')
    
    # Split into Train (In-Sample, approx 1 year) and Test
    # Assuming daily data, ~252 trading days per year
    train_size = 252 
    df_train = df.iloc[:train_size]
    
    # 2. Complete EM Calibration In-Sample
    calibrated_kf = calibrate_kalman(df_train['SNPS'], df_train['CDNS'])
    
    # 3. Apply dynamically over the entire period to visualize adaptation
    results = apply_dynamic_strategy(calibrated_kf, df['SNPS'], df['CDNS'], threshold=1.5)
    
    # 4. Calculate Backtest Metrics
    daily_returns = results['Daily_PnL'].values
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    
    print("\n--- Backtest Results ---")
    if std_ret > 0:
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    else:
        print("Sharpe Ratio: NaN (0 volatility, no trades or identical offset)")
        
    num_trades = np.count_nonzero(np.diff(results['Position'].values) != 0)
    print(f"Number of Trades / Position Changes: {num_trades}")
    print(f"Final Cumulative Return: {results['Cumulative_PnL'].iloc[-1]*100:.2f}%")
    
    # 5. Plot
    plot_results(results)
