"""
Volatility Surface Calibration - Day 6: Interactive Volatility Surface
Constructs an interactive volatility surface interpolator using 3D visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_surface_data():
    """
    Generate synthetic volatility surface data for testing.
    
    Returns:
        DataFrame with strikes, maturities, and implied volatilities
    """
    np.random.seed(42)
    
    data = []
    S0 = 100.0
    maturities = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    
    for T in maturities:
        strikes = np.linspace(60, 140, 25)
        atm_vol = 0.20 + 0.05 * np.sqrt(T)
        
        for K in strikes:
            moneyness = K / S0
            
            # Volatility smile with skew
            if moneyness < 1:
                vol = atm_vol + 0.4 * (1 - moneyness)**2 * 0.3
            else:
                vol = atm_vol + 0.2 * (moneyness - 1)**2 * 0.2
            
            vol += 0.02 * np.random.randn()
            vol = max(vol, 0.05)
            
            data.append({
                'strike': K,
                'maturity': T,
                'implied_vol': vol,
                'moneyness': K / S0
            })
    
    return pd.DataFrame(data)

def interpolate_surface(df, strike_grid, maturity_grid):
    """
    Interpolate the volatility surface using cubic interpolation.
    
    Args:
        df: DataFrame with strike, maturity, and implied volatility
        strike_grid: 2D array of strike values
        maturity_grid: 2D array of maturity values
    
    Returns:
        Interpolated volatility surface
    """
    points = df[['strike', 'maturity']].values
    values = df['implied_vol'].values
    
    # Interpolate using griddata
    vol_grid = griddata(points, values, (strike_grid, maturity_grid), 
                        method='cubic', fill_value=np.nan)
    
    return vol_grid

def calculate_greeks(S0, K, T, r, sigma):
    """
    Calculate option Greeks (Delta, Gamma, Vega) using Black-Scholes.
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Implied volatility
    
    Returns:
        Dictionary with Delta, Gamma, and Vega
    """
    from scipy.stats import norm
    
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0}
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta: sensitivity to underlying price
    delta = norm.cdf(d1)
    
    # Gamma: sensitivity of delta to underlying price
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    
    # Vega: sensitivity to volatility
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega
    }

def compute_surface_greeks(df, S0=100.0, r=0.05):
    """
    Compute Greeks for the entire volatility surface.
    
    Args:
        df: DataFrame with strike, maturity, and implied volatility
        S0: Current stock price
        r: Risk-free rate
    
    Returns:
        DataFrame with Greeks added
    """
    df = df.copy()
    
    deltas = []
    gammas = []
    vegas = []
    
    for idx, row in df.iterrows():
        K = row['strike']
        T = row['maturity']
        sigma = row['implied_vol']
        
        greeks = calculate_greeks(S0, K, T, r, sigma)
        deltas.append(greeks['delta'])
        gammas.append(greeks['gamma'])
        vegas.append(greeks['vega'])
    
    df['delta'] = deltas
    df['gamma'] = gammas
    df['vega'] = vegas
    
    return df

def plot_3d_surface(df):
    """
    Create a 3D visualization of the volatility surface.
    
    Args:
        df: DataFrame with strike, maturity, and implied volatility
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for surface
    strikes = np.linspace(df['strike'].min(), df['strike'].max(), 50)
    maturities = np.linspace(df['maturity'].min(), df['maturity'].max(), 50)
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    
    # Interpolate surface
    points = df[['strike', 'maturity']].values
    values = df['implied_vol'].values
    vol_grid = griddata(points, values, (strike_grid, maturity_grid), 
                        method='cubic', fill_value=np.nan)
    
    # Plot surface
    surf = ax.plot_surface(strike_grid, maturity_grid, vol_grid, 
                           cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    
    # Plot data points
    ax.scatter(df['strike'], df['maturity'], df['implied_vol'], 
               color='red', s=20, alpha=0.5, label='Market Data')
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Volatility Surface')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Implied Volatility')
    plt.tight_layout()
    plt.show()

def plot_greeks_surface(df):
    """
    Create 3D visualizations of option Greeks across the surface.
    
    Args:
        df: DataFrame with Greeks
    """
    fig = plt.figure(figsize=(18, 5))
    
    # Create grid for surface
    strikes = np.linspace(df['strike'].min(), df['strike'].max(), 50)
    maturities = np.linspace(df['maturity'].min(), df['maturity'].max(), 50)
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    
    greeks = ['delta', 'gamma', 'vega']
    titles = ['Delta (Δ)', 'Gamma (Γ)', 'Vega (V)']
    cmaps = ['RdBu', 'viridis', 'plasma']
    
    for i, (greek, title, cmap) in enumerate(zip(greeks, titles, cmaps)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Interpolate Greek surface
        points = df[['strike', 'maturity']].values
        values = df[greek].values
        greek_grid = griddata(points, values, (strike_grid, maturity_grid), 
                              method='cubic', fill_value=np.nan)
        
        # Plot surface
        surf = ax.plot_surface(strike_grid, maturity_grid, greek_grid, 
                               cmap=cmap, alpha=0.8, linewidth=0, antialiased=True)
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel(greek)
        ax.set_title(title)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.show()

def plot_volatility_smile(df):
    """
    Plot volatility smile at different maturities.
    
    Args:
        df: DataFrame with strike, maturity, and implied volatility
    """
    plt.figure(figsize=(12, 8))
    
    maturities = sorted(df['maturity'].unique())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(maturities)))
    
    for i, T in enumerate(maturities):
        if i % 2 == 0:
            subset = df[df['maturity'] == T]
            subset = subset.sort_values('strike')
            plt.plot(subset['strike'], subset['implied_vol'], 
                    'o-', label=f'T={T:.2f}', color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Smile Across Maturities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_greeks_comparison(df):
    """
    Plot Greeks comparison at a specific maturity.
    
    Args:
        df: DataFrame with Greeks
    """
    # Select a specific maturity
    target_T = df['maturity'].median()
    subset = df[np.abs(df['maturity'] - target_T) < 0.05]
    subset = subset.sort_values('strike')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    greeks = ['delta', 'gamma', 'vega']
    titles = ['Delta (Δ)', 'Gamma (Γ)', 'Vega (V)']
    ylabels = ['Δ', 'Γ', 'V']
    
    for i, (greek, title, ylabel) in enumerate(zip(greeks, titles, ylabels)):
        axes[i].plot(subset['strike'], subset[greek], 'o-', linewidth=2, markersize=6)
        axes[i].set_xlabel('Strike Price')
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(title)
        axes[i].grid(True)
    
    plt.suptitle(f'Option Greeks at Maturity T ≈ {target_T:.2f}')
    plt.tight_layout()
    plt.show()

def generate_interactive_vol_surface():
    """
    Generate an interactive volatility surface using synthetic data.
    
    Returns:
        DataFrame with volatility surface data
    """
    data = []
    S0 = 100.0
    
    # Generate strikes and maturities
    strikes = np.linspace(60, 140, 30)
    maturities = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    
    for T in maturities:
        atm_vol = 0.20 + 0.05 * np.sqrt(T)
        
        for K in strikes:
            moneyness = K / S0
            
            # Volatility smile with skew
            if moneyness < 1:
                vol = atm_vol + 0.4 * (1 - moneyness)**2 * 0.3
            else:
                vol = atm_vol + 0.2 * (moneyness - 1)**2 * 0.2
            
            # Add small noise
            vol += 0.01 * np.random.randn()
            vol = max(vol, 0.05)
            
            data.append({
                'strike': K,
                'maturity': T,
                'implied_vol': vol,
                'moneyness': K / S0
            })
    
    return pd.DataFrame(data)

def main():
    """
    Run interactive volatility surface pipeline.
    """
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION - DAY 6: INTERACTIVE VOLATILITY SURFACE")
    print("=" * 70)
    
    # Generate synthetic surface data
    df = generate_interactive_vol_surface()
    print(f"Generated {len(df)} data points")
    
    # Compute Greeks
    print("\nComputing option Greeks...")
    df = compute_surface_greeks(df)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("VOLATILITY SURFACE SUMMARY")
    print("=" * 70)
    print(f"Number of data points: {len(df)}")
    print(f"Strike range: {df['strike'].min():.2f} - {df['strike'].max():.2f}")
    print(f"Maturity range: {df['maturity'].min():.2f} - {df['maturity'].max():.2f}")
    print(f"Implied vol range: {df['implied_vol'].min():.4f} - {df['implied_vol'].max():.4f}")
    print(f"Mean implied vol: {df['implied_vol'].mean():.4f}")
    print(f"\nGreeks summary:")
    print(f"  Delta range: {df['delta'].min():.4f} - {df['delta'].max():.4f}")
    print(f"  Gamma range: {df['gamma'].min():.4f} - {df['gamma'].max():.4f}")
    print(f"  Vega range:  {df['vega'].min():.4f} - {df['vega'].max():.4f}")
    
    # Plot volatility surface
    print("\n" + "=" * 70)
    print("DISPLAYING VOLATILITY SURFACE")
    print("=" * 70)
    plot_3d_surface(df)
    
    # Plot volatility smile
    print("\n" + "=" * 70)
    print("DISPLAYING VOLATILITY SMILE")
    print("=" * 70)
    plot_volatility_smile(df)
    
    # Plot Greeks
    print("\n" + "=" * 70)
    print("DISPLAYING OPTION GREEKS")
    print("=" * 70)
    plot_greeks_surface(df)
    plot_greeks_comparison(df)
    
    print("\n" + "=" * 70)
    print("Interactive volatility surface construction complete.")
    print("Ready for Day 7: Risk Sensitivities.")
    print("=" * 70)

if __name__ == "__main__":
    main()
