"""
Volatility Surface Calibration - Day 1: Data Collection
Collects option chain data via Yahoo Finance API for multiple strike-maturity pairs.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

# Set random seed for reproducibility
np.random.seed(42)

def get_stock_price(ticker: str) -> float:
    """
    Get current stock price for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'SPY')
    
    Returns:
        Current stock price
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    return hist['Close'].iloc[-1]

def get_option_chain(ticker: str, expiration_date: str) -> pd.DataFrame:
    """
    Fetch option chain data for a given ticker and expiration date.
    
    Args:
        ticker: Stock ticker symbol
        expiration_date: Expiration date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with option data including strikes and prices
    """
    stock = yf.Ticker(ticker)
    
    try:
        # Get option chain for the specified expiration
        opt_chain = stock.option_chain(expiration_date)
        
        # Combine calls and puts
        calls = opt_chain.calls
        calls['option_type'] = 'call'
        puts = opt_chain.puts
        puts['option_type'] = 'put'
        
        chain = pd.concat([calls, puts], ignore_index=True)
        chain['expiration'] = expiration_date
        
        return chain
    except Exception as e:
        print(f"Error fetching options for expiration {expiration_date}: {e}")
        return pd.DataFrame()

def get_available_expirations(ticker: str) -> list:
    """
    Get list of available option expiration dates for a ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        List of expiration dates as strings
    """
    stock = yf.Ticker(ticker)
    expirations = stock.options
    return expirations

def collect_option_data(ticker: str, num_expirations: int = 10, min_strikes: int = 20) -> pd.DataFrame:
    """
    Collect option chain data across multiple expirations.
    
    Args:
        ticker: Stock ticker symbol
        num_expirations: Number of expiration dates to collect
        min_strikes: Minimum number of strikes required per expiration
    
    Returns:
        DataFrame with all collected option data
    """
    print(f"\nCollecting option data for {ticker}...")
    
    # Get current stock price
    current_price = get_stock_price(ticker)
    print(f"Current {ticker} price: ${current_price:.2f}")
    
    # Get available expirations
    expirations = get_available_expirations(ticker)
    print(f"Available expirations: {len(expirations)}")
    
    # Select expirations that are at least 7 days out
    today = datetime.now().date()
    valid_expirations = []
    for exp in expirations:
        exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
        if (exp_date - today).days >= 7:
            valid_expirations.append(exp)
    
    # Take the first num_expirations valid expirations
    selected_expirations = valid_expirations[:num_expirations]
    print(f"Selected {len(selected_expirations)} expirations")
    
    all_data = []
    
    for exp_date in selected_expirations:
        print(f"  Fetching {exp_date}...")
        
        chain = get_option_chain(ticker, exp_date)
        
        if not chain.empty:
            # Add time to maturity in years
            exp_date_obj = datetime.strptime(exp_date, '%Y-%m-%d').date()
            ttm = (exp_date_obj - today).days / 365.0
            chain['time_to_maturity'] = ttm
            
            # Add current price for reference
            chain['current_price'] = current_price
            
            all_data.append(chain)
            
            print(f"    Got {len(chain)} options")
        
        # Rate limit to avoid API throttling
        time.sleep(0.5)
    
    if not all_data:
        print("No data collected!")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter for reasonable strikes (50% to 150% of current price)
    combined_df = combined_df[
        (combined_df['strike'] >= 0.5 * current_price) &
        (combined_df['strike'] <= 1.5 * current_price)
    ]
    
    # Filter for reasonable bid-ask spreads
    combined_df['spread'] = combined_df['ask'] - combined_df['bid']
    combined_df = combined_df[combined_df['spread'] > 0]
    
    # Filter for options with volume
    combined_df = combined_df[combined_df['volume'] > 0]
    
    print(f"\nTotal options collected: {len(combined_df)}")
    
    return combined_df

def generate_synthetic_option_data(n_strikes: int = 200, n_expirations: int = 5) -> pd.DataFrame:
    """
    Generate synthetic option data for testing and development.
    This is useful when the Yahoo Finance API is not available.
    
    Args:
        n_strikes: Number of strike-maturity pairs to generate
        n_expirations: Number of different expirations
    
    Returns:
        DataFrame with synthetic option data
    """
    # Parameters
    S0 = 100.0  # Current stock price
    r = 0.05  # Risk-free rate
    sigma_ATM = 0.25  # At-the-money volatility
    
    np.random.seed(42)
    
    data = []
    
    for exp_idx in range(n_expirations):
        T = 0.1 + exp_idx * 0.3  # Time to maturity: 0.1 to 1.3 years
        
        # Generate strikes around the current price
        strikes = np.linspace(0.6 * S0, 1.4 * S0, n_strikes // n_expirations)
        
        # For each strike, generate implied volatility with a smile
        for K in strikes:
            # Volatility smile: higher for OTM options
            moneyness = K / S0
            
            # Base volatility (ATM)
            vol = sigma_ATM
            
            # Add smile: higher vol for deep OTM and ITM
            smile_factor = 0.3 * (moneyness - 1) ** 2
            vol = vol + smile_factor
            
            # Add some noise
            vol *= (1 + 0.05 * np.random.randn())
            vol = max(vol, 0.05)
            
            # Generate market price using Black-Scholes
            d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            from scipy.stats import norm
            call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            
            # Simulate mid prices with bid-ask spread
            spread = 0.005 * call_price  # 0.5% spread for calls
            call_bid = call_price - spread/2
            call_ask = call_price + spread/2
            
            # Store data
            data.append({
                'strike': K,
                'expiration': f'2026-{exp_idx+1:02d}-{15+exp_idx*30:02d}',
                'time_to_maturity': T,
                'implied_volatility': vol,
                'call_price': call_price,
                'call_bid': call_bid,
                'call_ask': call_ask,
                'put_price': put_price,
                'moneyness': moneyness,
                'is_synthetic': True
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} synthetic option data points")
    return df

def save_data(df: pd.DataFrame, ticker: str = 'SPY'):
    """
    Save the collected data to a CSV file.
    """
    if df.empty:
        print("No data to save!")
        return
    
    filename = f"data/option_data_{ticker}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    # Also save a summary
    summary = {
        'ticker': ticker,
        'total_options': len(df),
        'expirations': sorted(df['expiration'].unique().tolist()),
        'strike_min': df['strike'].min(),
        'strike_max': df['strike'].max(),
        'time_to_maturity_min': df['time_to_maturity'].min(),
        'time_to_maturity_max': df['time_to_maturity'].max()
    }
    
    print("\nData Summary:")
    print(f"  Total options: {summary['total_options']}")
    print(f"  Number of expirations: {len(summary['expirations'])}")
    print(f"  Strike range: {summary['strike_min']:.2f} - {summary['strike_max']:.2f}")
    print(f"  Maturity range: {summary['time_to_maturity_min']:.2f} - {summary['time_to_maturity_max']:.2f} years")

def main():
    """
    Run data collection pipeline.
    """
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION - DAY 1: DATA COLLECTION")
    print("=" * 70)
    
    # Parameters
    ticker = "SPY"  # S&P 500 ETF
    num_expirations = 5
    
    print(f"\nCollecting data for {ticker}...")
    
    # Try to fetch real data first
    try:
        df = collect_option_data(ticker, num_expirations=num_expirations)
        if df.empty or len(df) < 50:
            print("Real data collection failed or returned insufficient data.")
            print("Generating synthetic data instead...")
            df = generate_synthetic_option_data()
    except Exception as e:
        print(f"Error fetching real data: {e}")
        print("Generating synthetic data instead...")
        df = generate_synthetic_option_data()
    
    if df.empty:
        print("No data available. Exiting.")
        return
    
    # Save the data
    save_data(df, ticker)
    
    # Print sample data
    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("Data collection complete. Ready for Day 2: Implied Volatility Extraction.")
    print("=" * 70)

if __name__ == "__main__":
    main()
