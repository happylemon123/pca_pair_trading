import yfinance as yf
import pandas as pd
import os
import datetime

def fetch_data(tickers, start_date="2020-01-01", end_date="2023-01-01"):
    """
    Fetches historical adjusted close prices for a list of tickers.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    try:
        # Download all data first
        raw_data = yf.download(tickers, start=start_date, end=end_date)
        
        # Check if empty
        if raw_data.empty:
            print("No data downloaded. Please check your tickers or internet connection.")
            return None
            
        print("Raw data columns:", raw_data.columns)
        
        # Extract Adj Close
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data.columns:
            print("Warning: 'Adj Close' not found, using 'Close' instead.")
            data = raw_data['Close']
        else:
            # Maybe it's multi-index and we need to check levels?
            # Or maybe single ticker?
            print("Could not find 'Adj Close' or 'Close'. Returning full data for inspection.")
            return raw_data

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    # Drop columns with all NaNs
    data.dropna(axis=1, how='all', inplace=True)
    
    # Fill missing values (forward fill then backward fill)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    
    print(f"Successfully downloaded data with shape: {data.shape}")
    return data

def save_data(data, filename="data/stock_prices.csv"):
    """
    Saves the dataframe to a CSV file.
    """
    if data is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data.to_csv(filename)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    # Top holdings of XLK (Technology Select Sector SPDR Fund) as a proxy for the tech sector
    tech_tickers = [
        "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "QCOM", "INTC", "TXN",
        "IBM", "AMAT", "NOW", "LRCX", "ADI", "MU", "KLAC", "SNPS", "CDNS", "PANW"
    ]
    
    # Fetch data
    end = datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date.today() - datetime.timedelta(days=3*365)).strftime("%Y-%m-%d")
    
    df = fetch_data(tech_tickers, start_date=start, end_date=end)
    
    # Save to data directory
    save_data(df)
