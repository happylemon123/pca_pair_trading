import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_market_data(n_days=1000):
    """
    Simulates a market with two regimes:
    1. Bull Market: Positive returns, Low Volatility
    2. Bear Market: Negative returns, High Volatility
    """
    # Regime 0: Bull (Mean=0.1%, Vol=1%)
    bull_mean = 0.001
    bull_vol = 0.01
    
    # Regime 1: Bear (Mean=-0.2%, Vol=3%)
    bear_mean = -0.002
    bear_vol = 0.03
    
    # Generate regime sequence (Markov Chain)
    # 95% chance to stay in Bull, 90% chance to stay in Bear
    regimes = [0]
    returns = []
    
    for i in range(1, n_days):
        current_regime = regimes[-1]
        if current_regime == 0:
            next_regime = 0 if np.random.rand() < 0.95 else 1
        else:
            next_regime = 1 if np.random.rand() < 0.90 else 0
        regimes.append(next_regime)
        
        # Generate return based on regime
        if next_regime == 0:
            ret = np.random.normal(bull_mean, bull_vol)
        else:
            ret = np.random.normal(bear_mean, bear_vol)
        returns.append(ret)
        
    dates = pd.date_range(start='2020-01-01', periods=n_days)
    df = pd.DataFrame({'Date': dates, 'Returns': returns, 'True_Regime': regimes})
    df['Price'] = 100 * (1 + df['Returns']).cumprod()
    return df

def fit_hmm(df):
    """
    Fits a Hidden Markov Model to detect regimes based on Returns.
    """
    print("Training Hidden Markov Model...")
    
    # Reshape for HMM (requires 2D array)
    X = df['Returns'].values.reshape(-1, 1)
    
    # Initialize HMM with 2 components (Bull/Bear)
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    hmm.fit(X)
    
    # Predict the hidden states
    df['Predicted_Regime'] = hmm.predict(X)
    
    # Check which state is which (State 0 isn't always Bull)
    # We assume the state with HIGHER volatility is the Bear market
    vol_0 = df[df['Predicted_Regime'] == 0]['Returns'].std()
    vol_1 = df[df['Predicted_Regime'] == 1]['Returns'].std()
    
    print(f"State 0 Volatility: {vol_0:.4f}")
    print(f"State 1 Volatility: {vol_1:.4f}")
    
    if vol_1 > vol_0:
        print("State 1 is High Volatility (Bear).")
        df['Regime_Label'] = df['Predicted_Regime'].map({0: 'Bull/Calm', 1: 'Bear/Crisis'})
    else:
        print("State 0 is High Volatility (Bear).")
        df['Regime_Label'] = df['Predicted_Regime'].map({1: 'Bull/Calm', 0: 'Bear/Crisis'})
        
    return df

def plot_regimes(df):
    """
    Plots the stock price colored by the detected regime.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Bull markets in Green
    bull_data = df[df['Regime_Label'] == 'Bull/Calm']
    plt.scatter(bull_data['Date'], bull_data['Price'], c='green', s=10, label='Bull/Calm', alpha=0.6)
    
    # Plot Bear markets in Red
    bear_data = df[df['Regime_Label'] == 'Bear/Crisis']
    plt.scatter(bear_data['Date'], bear_data['Price'], c='red', s=10, label='Bear/Crisis', alpha=0.6)
    
    plt.title('Market Regime Detection using Hidden Markov Models (HMM)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('market_regimes.png')
    print("Plot saved to market_regimes.png")

if __name__ == "__main__":
    # 1. Generate Dummy Data
    print("Generating simulated market data...")
    market_df = generate_market_data()
    
    # 2. Fit HMM
    market_df = fit_hmm(market_df)
    
    # 3. Plot Results
    plot_regimes(market_df)
    
    print("\nAnalysis Complete.")
    print(market_df[['Date', 'Price', 'Regime_Label']].head())
