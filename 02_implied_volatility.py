"""
Volatility Surface Calibration - Day 2: Implied Volatility Extraction
Computes implied volatilities from option market prices using the Black-Scholes model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def black_scholes_call(S0: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Black-Scholes formula for a European call option.
    
    Args:
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        T: Time to maturity
        sigma: Volatility
    
    Returns:
        Call option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S0: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Black-Scholes formula for a European put option.
    
    Args:
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        T: Time to maturity
        sigma: Volatility
    
    Returns:
        Put option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def implied_volatility_call(price: float, S0: float, K: float, r: float, T: float) -> float:
    """
    Compute implied volatility for a call option using Brent's method.
    
    Args:
        price: Market price of the call option
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        T: Time to maturity
    
    Returns:
        Implied volatility
    """
    if price <= 0:
        return np.nan
    
    # Use Brent's method to find the root
    def objective(sigma):
        return black_scholes_call(S0, K, r, T, sigma) - price
    
    try:
        # Bracket search for implied volatility
        # Lower bound: 0.001, Upper bound: 2.0 (200% volatility)
        iv = brentq(objective, 0.001, 2.0, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        return np.nan

def implied_volatility_put(price: float, S0: float, K: float, r: float, T: float) -> float:
    """
    Compute implied volatility for a put option using Brent's method.
    
    Args:
        price: Market price of the put option
        S0: Current stock price
        K: Strike price
        r: Risk-free rate
        T: Time to maturity
    
    Returns:
        Implied volatility
    """
    if price <= 0:
        return np.nan
    
    def objective(sigma):
        return black_scholes_put(S0, K, r, T, sigma) - price
    
    try:
        iv = brentq(objective, 0.001, 2.0, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        return np.nan

def compute_implied_volatilities(df: pd.DataFrame, r: float = 0.05) -> pd.DataFrame:
    """
    Compute implied volatilities for all options in the dataset.
    
    Args:
        df: DataFrame with option data
        r: Risk-free rate
    
    Returns:
        DataFrame with implied volatilities added
    """
    df = df.copy()
    
    # Use mid price when available
    if 'call_price' in df.columns and df['call_price'].notna().all():
        call_prices = df['call_price']
    elif 'call_bid' in df.columns and 'call_ask' in df.columns:
        call_prices = (df['call_bid'] + df['call_ask']) / 2
    elif 'lastPrice' in df.columns:
        call_prices = df['lastPrice']
    else:
        print("Warning: No price data found. Using synthetic values.")
        # Use synthetic call prices from the data collection step
        call_prices = df['call_price'] if 'call_price' in df.columns else None
    
    # Compute implied volatility for each option
    implied_vols = []
    option_types = []
    
    for idx, row in df.iterrows():
        S0 = row['current_price'] if 'current_price' in row else 100.0
        K = row['strike']
        T = row['time_to_maturity']
        
        # Use mid price
        if 'call_price' in row and row['call_price'] is not None:
            price = row['call_price']
        elif 'call_bid' in row and 'call_ask' in row:
            price = (row['call_bid'] + row['call_ask']) / 2
        else:
            # Fallback: use synthetic calculation from implied volatility in data
            if 'implied_volatility' in row and row['implied_volatility'] is not None:
                implied_vols.append(row['implied_volatility'])
                option_types.append('call')
                continue
            else:
                implied_vols.append(np.nan)
                option_types.append('call')
                continue
        
        # Determine option type
        if 'option_type' in row:
            opt_type = row['option_type']
        else:
            opt_type = 'call'  # Default to call if not specified
        
        if opt_type == 'call':
            iv = implied_volatility_call(price, S0, K, r, T)
        else:
            iv = implied_volatility_put(price, S0, K, r, T)
        
        implied_vols.append(iv)
        option_types.append(opt_type)
    
    df['implied_volatility'] = implied_vols
    df['option_type'] = option_types
    
    return df

def clean_implied_volatilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter implied volatility data.
    
    Args:
        df: DataFrame with implied volatilities
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove NaN implied volatilities
    df = df.dropna(subset=['implied_volatility'])
    
    # Filter unreasonable implied volatilities (0.01 to 2.0)
    df = df[(df['implied_volatility'] > 0.01) & (df['implied_volatility'] < 2.0)]
    
    # Calculate moneyness
    if 'current_price' in df.columns:
        df['moneyness'] = df['strike'] / df['current_price']
    else:
        df['moneyness'] = 1.0
    
    return df

def compute_volatility_smile(df: pd.DataFrame) -> dict:
    """
    Compute volatility smile data grouped by time to maturity.
    
    Args:
        df: DataFrame with cleaned implied volatilities
    
    Returns:
        Dictionary with smile data for each maturity
    """
    smiles = {}
    
    # Group by time to maturity (binned)
    df['maturity_bin'] = pd.cut(df['time_to_maturity'], bins=5)
    
    for name, group in df.groupby('maturity_bin'):
        # Sort by moneyness
        group = group.sort_values('moneyness')
        smiles[str(name)] = {
            'moneyness': group['moneyness'].values,
            'implied_volatility': group['implied_volatility'].values,
            'strikes': group['strike'].values
        }
    
    return smiles

def plot_volatility_smile(df: pd.DataFrame) -> None:
    """
    Plot the volatility smile for each maturity.
    
    Args:
        df: DataFrame with implied volatilities
    """
    plt.figure(figsize=(12, 8))
    
    # Group by time to maturity
    maturities = sorted(df['time_to_maturity'].unique())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(maturities)))
    
    for i, T in enumerate(maturities):
        if i % 3 == 0:  # Plot every 3rd maturity to avoid clutter
            subset = df[df['time_to_maturity'] == T]
            subset = subset.sort_values('moneyness')
            plt.plot(subset['moneyness'], subset['implied_volatility'], 
                    'o-', label=f'T={T:.2f}', color=colors[i], alpha=0.7)
    
    plt.xlabel('Moneyness (K/S0)')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Smile Across Maturities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_synthetic_implied_volatilities() -> pd.DataFrame:
    """
    Generate synthetic implied volatility data for testing.
    """
    np.random.seed(42)
    
    S0 = 100.0
    n_strikes = 40
    maturities = [0.1, 0.3, 0.5, 0.8, 1.2, 1.8]
    
    data = []
    
    for T in maturities:
        strikes = np.linspace(60, 140, n_strikes)
        atm_vol = 0.20 + 0.05 * T
        
        for K in strikes:
            moneyness = K / S0
            
            # Simulate volatility smile
            if moneyness < 1:
                vol = atm_vol + 0.5 * (1 - moneyness)**2 * 0.3
            else:
                vol = atm_vol + 0.3 * (moneyness - 1)**2 * 0.2
            
            # Add some noise
            vol += 0.02 * np.random.randn()
            vol = max(vol, 0.05)
            
            data.append({
                'strike': K,
                'current_price': S0,
                'time_to_maturity': T,
                'moneyness': moneyness,
                'implied_volatility': vol,
                'option_type': 'call'
            })
    
    df = pd.DataFrame(data)
    return df

def main():
    """
    Run implied volatility extraction pipeline.
    """
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION - DAY 2: IMPLIED VOLATILITY EXTRACTION")
    print("=" * 70)
    
    # Try to load real data, fallback to synthetic
    try:
        df = pd.read_csv('data/option_data_SPY.csv')
        print(f"Loaded {len(df)} options from data file")
        
        # Check if we have the necessary columns
        if 'implied_volatility' not in df.columns:
            print("Computing implied volatilities...")
            df = compute_implied_volatilities(df)
        else:
            print("Implied volatilities already present in data")
        
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("No data file found. Generating synthetic data...")
        df = generate_synthetic_implied_volatilities()
    
    # Clean the data
    df_cleaned = clean_implied_volatilities(df)
    print(f"\nCleaned data: {len(df_cleaned)} options remaining")
    
    # Compute volatility smile
    smiles = compute_volatility_smile(df_cleaned)
    print(f"\nVolatility smile computed for {len(smiles)} maturity bins")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("IMPLIED VOLATILITY SUMMARY")
    print("=" * 70)
    print(f"Total options processed: {len(df)}")
    print(f"Valid implied volatilities: {len(df_cleaned)}")
    print(f"Mean implied volatility: {df_cleaned['implied_volatility'].mean():.4f}")
    print(f"Std implied volatility: {df_cleaned['implied_volatility'].std():.4f}")
    print(f"Min implied volatility: {df_cleaned['implied_volatility'].min():.4f}")
    print(f"Max implied volatility: {df_cleaned['implied_volatility'].max():.4f}")
    
    # Plot the volatility smile
    print("\n" + "=" * 70)
    print("DISPLAYING VOLATILITY SMILE")
    print("=" * 70)
    plot_volatility_smile(df_cleaned)
    
    print("\n" + "=" * 70)
    print("Implied volatility extraction complete.")
    print("Ready for Day 3: SABR Model Calibration.")
    print("=" * 70)

if __name__ == "__main__":
    main()
