import numpy as np
import pandas as pd
import statsmodels.api as sm

def calculate_ou_parameters(spread_series):
    """
    Calculates Kappa (speed of mean reversion) and Theta (long-term mean)
    using an AR(1) Linear Regression on the spread.
    """
    # 1. Create the dependent variable: Delta Spread (yt - y_{t-1})
    delta_spread = spread_series.diff().dropna()
    
    # 2. Create the independent variable: Previous period's spread (y_{t-1})
    lagged_spread = spread_series.shift(1).dropna()
    
    # Align
    df = pd.DataFrame({'delta_y': delta_spread, 'y_lagged': lagged_spread})
    
    # 3. Add a constant for the intercept (c)
    X = sm.add_constant(df['y_lagged'])
    Y = df['delta_y']
    
    # 4. Run Ordinary Least Squares (OLS) Linear Regression
    model = sm.OLS(Y, X).fit()
    
    # 5. Extract our parameters from the regression results
    c = model.params['const']
    b = model.params['y_lagged']
    
    # 6. Translate Linear Regression (Discrete) to OU Process (Continuous)
    kappa = -b
    theta = c / kappa if kappa != 0 else np.nan
    half_life = np.log(2) / kappa if kappa > 0 else np.nan
    
    print("--- OLS Regression Results ---")
    print(f"Intercept (c) : {c:.6f}")
    print(f"Slope (b)     : {b:.6f}")
    print("\n--- Ornstein-Uhlenbeck Parameters ---")
    print(f"Kappa (Speed) : {kappa:.6f}")
    print(f"Theta (Mean)  : {theta:.6f}")
    print(f"Half-Life     : {half_life:.2f} days")
    
    return kappa, theta, half_life

if __name__ == "__main__":
    np.random.seed(42)
    fake_spread = [0]
    for _ in range(1000):
        next_val = fake_spread[-1] + 0.1 * (5.0 - fake_spread[-1]) + np.random.normal(0, 1)
        fake_spread.append(next_val)
        
    print("Testing on Synthetic Spread (True Theta=5.0, True Kappa=0.1)")
    calculate_ou_parameters(pd.Series(fake_spread))
