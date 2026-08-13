"""
Volatility Surface Calibration - Day 5: Model Comparison
Compares SABR and Heston model fits and analyzes their performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Import functions from previous days
from sabr_calibration import sabr_implied_volatility
from heston_calibration import heston_implied_volatility

def compare_models(df, sabr_params, heston_params):
    """
    Compare SABR and Heston model fits against market data.
    
    Args:
        df: DataFrame with market implied volatilities
        sabr_params: Dictionary with SABR parameters
        heston_params: Dictionary with Heston parameters
    
    Returns:
        Dictionary with comparison metrics
    """
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    S0 = 100.0
    r = 0.05
    
    # Calculate errors for each model
    sabr_errors = []
    heston_errors = []
    strikes = []
    maturities = []
    
    for idx, row in df.iterrows():
        K = row['strike']
        T = row['time_to_maturity']
        market_vol = row['implied_volatility']
        
        # SABR model
        sabr_vol = sabr_implied_volatility(
            S0, K, T, 
            sabr_params['alpha'], 
            sabr_params['beta'], 
            sabr_params['rho'], 
            sabr_params['nu']
        )
        sabr_error = sabr_vol - market_vol
        sabr_errors.append(sabr_error)
        
        # Heston model
        heston_vol = heston_implied_volatility(
            S0, K, T, r,
            heston_params['kappa'],
            heston_params['theta'],
            heston_params['sigma'],
            heston_params['rho'],
            heston_params['v0']
        )
        heston_error = heston_vol - market_vol
        heston_errors.append(heston_error)
        
        strikes.append(K)
        maturities.append(T)
    
    # Convert to arrays
    sabr_errors = np.array(sabr_errors)
    heston_errors = np.array(heston_errors)
    
    # Calculate statistics
    sabr_rmse = np.sqrt(np.mean(sabr_errors**2))
    heston_rmse = np.sqrt(np.mean(heston_errors**2))
    sabr_mae = np.mean(np.abs(sabr_errors))
    heston_mae = np.mean(np.abs(heston_errors))
    sabr_bias = np.mean(sabr_errors)
    heston_bias = np.mean(heston_errors)
    sabr_std = np.std(sabr_errors)
    heston_std = np.std(heston_errors)
    
    print("\nMODEL PERFORMANCE METRICS:")
    print("-" * 70)
    print(f"{'Metric':<20} {'SABR':<15} {'Heston':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'RMSE':<20} {sabr_rmse:<15.6f} {heston_rmse:<15.6f} {sabr_rmse - heston_rmse:<15.6f}")
    print(f"{'MAE':<20} {sabr_mae:<15.6f} {heston_mae:<15.6f} {sabr_mae - heston_mae:<15.6f}")
    print(f"{'Bias':<20} {sabr_bias:<15.6f} {heston_bias:<15.6f} {sabr_bias - heston_bias:<15.6f}")
    print(f"{'Std Dev':<20} {sabr_std:<15.6f} {heston_std:<15.6f} {sabr_std - heston_std:<15.6f}")
    
    # Determine which model performs better
    if sabr_rmse < heston_rmse:
        print("\nSABR model has lower RMSE and performs better on this dataset.")
    elif heston_rmse < sabr_rmse:
        print("\nHeston model has lower RMSE and performs better on this dataset.")
    else:
        print("\nBoth models perform equally well on this dataset.")
    
    return {
        'sabr': {
            'errors': sabr_errors,
            'rmse': sabr_rmse,
            'mae': sabr_mae,
            'bias': sabr_bias,
            'std': sabr_std
        },
        'heston': {
            'errors': heston_errors,
            'rmse': heston_rmse,
            'mae': heston_mae,
            'bias': heston_bias,
            'std': heston_std
        },
        'strikes': strikes,
        'maturities': maturities
    }

def plot_error_comparison(comparison_results):
    """
    Plot error comparison between SABR and Heston models.
    
    Args:
        comparison_results: Dictionary from compare_models
    """
    sabr_errors = comparison_results['sabr']['errors']
    heston_errors = comparison_results['heston']['errors']
    strikes = comparison_results['strikes']
    maturities = comparison_results['maturities']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error vs Strike
    axes[0, 0].scatter(strikes, sabr_errors, alpha=0.6, label='SABR', color='blue')
    axes[0, 0].scatter(strikes, heston_errors, alpha=0.6, label='Heston', color='red')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 0].set_xlabel('Strike Price')
    axes[0, 0].set_ylabel('Error (Model - Market)')
    axes[0, 0].set_title('Calibration Error vs Strike')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Error vs Maturity
    axes[0, 1].scatter(maturities, sabr_errors, alpha=0.6, label='SABR', color='blue')
    axes[0, 1].scatter(maturities, heston_errors, alpha=0.6, label='Heston', color='red')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 1].set_xlabel('Time to Maturity')
    axes[0, 1].set_ylabel('Error (Model - Market)')
    axes[0, 1].set_title('Calibration Error vs Maturity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Error Distribution
    axes[1, 0].hist(sabr_errors, bins=30, alpha=0.5, label='SABR', color='blue')
    axes[1, 0].hist(heston_errors, bins=30, alpha=0.5, label='Heston', color='red')
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Cumulative Error
    sabr_sorted = np.sort(np.abs(sabr_errors))
    heston_sorted = np.sort(np.abs(heston_errors))
    x = np.linspace(0, 1, len(sabr_sorted))
    
    axes[1, 1].plot(x, sabr_sorted, label='SABR', color='blue', linewidth=2)
    axes[1, 1].plot(x, heston_sorted, label='Heston', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Percentile')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Cumulative Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_volatility_surface_comparison(df, sabr_params, heston_params):
    """
    Plot and compare volatility surfaces from both models.
    
    Args:
        df: DataFrame with market implied volatilities
        sabr_params: Dictionary with SABR parameters
        heston_params: Dictionary with Heston parameters
    """
    S0 = 100.0
    r = 0.05
    
    fig = plt.figure(figsize=(15, 10))
    
    # Get unique maturities
    maturities = sorted(df['time_to_maturity'].unique())
    
    for i, T in enumerate(maturities):
        if i >= 3:  # Plot first 3 maturities to avoid clutter
            break
        
        subset = df[df['time_to_maturity'] == T]
        subset = subset.sort_values('strike')
        
        strikes = subset['strike'].values
        market_vols = subset['implied_volatility'].values
        
        # Compute model vols for the same strikes
        sabr_vols = []
        heston_vols = []
        for K in strikes:
            sabr_vol = sabr_implied_volatility(
                S0, K, T,
                sabr_params['alpha'],
                sabr_params['beta'],
                sabr_params['rho'],
                sabr_params['nu']
            )
            sabr_vols.append(sabr_vol)
            
            heston_vol = heston_implied_volatility(
                S0, K, T, r,
                heston_params['kappa'],
                heston_params['theta'],
                heston_params['sigma'],
                heston_params['rho'],
                heston_params['v0']
            )
            heston_vols.append(heston_vol)
        
        # Plot
        ax = fig.add_subplot(1, len(maturities), i+1)
        ax.scatter(strikes, market_vols, label='Market', color='black', s=30, zorder=5)
        ax.plot(strikes, sabr_vols, label='SABR', color='blue', linewidth=2)
        ax.plot(strikes, heston_vols, label='Heston', color='red', linewidth=2, linestyle='--')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'T = {T:.2f}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def generate_synthetic_vol_surface():
    """
    Generate synthetic volatility surface data for testing.
    """
    np.random.seed(42)
    
    data = []
    S0 = 100.0
    maturities = [0.1, 0.3, 0.5, 0.8, 1.2]
    
    # True parameters for generating data
    true_sabr = {'alpha': 0.25, 'beta': 0.5, 'rho': -0.3, 'nu': 0.4}
    
    for T in maturities:
        strikes = np.linspace(60, 140, 20)
        for K in strikes:
            vol = sabr_implied_volatility(
                S0, K, T,
                true_sabr['alpha'],
                true_sabr['beta'],
                true_sabr['rho'],
                true_sabr['nu']
            )
            # Add noise to simulate market data
            vol += 0.02 * np.random.randn()
            vol = max(vol, 0.05)
            data.append({
                'strike': K,
                'time_to_maturity': T,
                'implied_volatility': vol
            })
    
    return pd.DataFrame(data)

def main():
    """
    Run model comparison pipeline.
    """
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION - DAY 5: MODEL COMPARISON")
    print("=" * 70)
    
    # Generate synthetic data
    df = generate_synthetic_vol_surface()
    print(f"Generated {len(df)} data points for testing")
    
    # For demonstration, we need to import the calibration parameters
    # Since we can't import from previous files directly, we'll use synthetic params
    
    # Simulated SABR parameters (from calibration)
    sabr_params = {
        'alpha': 0.24,
        'beta': 0.48,
        'rho': -0.32,
        'nu': 0.38
    }
    
    # Simulated Heston parameters (from calibration)
    heston_params = {
        'kappa': 1.8,
        'theta': 0.038,
        'sigma': 0.28,
        'rho': -0.45,
        'v0': 0.035
    }
    
    print("\nSABR Parameters:")
    print(f"  alpha: {sabr_params['alpha']:.4f}")
    print(f"  beta:  {sabr_params['beta']:.4f}")
    print(f"  rho:   {sabr_params['rho']:.4f}")
    print(f"  nu:    {sabr_params['nu']:.4f}")
    
    print("\nHeston Parameters:")
    print(f"  kappa: {heston_params['kappa']:.4f}")
    print(f"  theta: {heston_params['theta']:.4f}")
    print(f"  sigma: {heston_params['sigma']:.4f}")
    print(f"  rho:   {heston_params['rho']:.4f}")
    print(f"  v0:    {heston_params['v0']:.4f}")
    
    # Compare models
    comparison_results = compare_models(df, sabr_params, heston_params)
    
    # Plot error comparison
    print("\n" + "=" * 70)
    print("DISPLAYING ERROR COMPARISON")
    print("=" * 70)
    plot_error_comparison(comparison_results)
    
    # Plot volatility surface comparison
    print("\n" + "=" * 70)
    print("DISPLAYING VOLATILITY SURFACE COMPARISON")
    print("=" * 70)
    plot_volatility_surface_comparison(df, sabr_params, heston_params)
    
    print("\n" + "=" * 70)
    print("Model comparison complete.")
    print("Ready for Day 6: Interactive Volatility Surface.")
    print("=" * 70)

if __name__ == "__main__":
    main()
