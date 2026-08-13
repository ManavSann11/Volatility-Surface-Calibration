"""
Volatility Surface Calibration - Day 3: SABR Model Calibration
Implements the SABR stochastic volatility model and calibrates it to market data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def sabr_implied_volatility(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Compute SABR implied volatility using the Hagan et al. approximation.
    
    The SABR model dynamics:
    dF = alpha * F^beta * dW1
    dalpha = nu * alpha * dW2
    dW1 * dW2 = rho * dt
    
    Args:
        F: Forward price
        K: Strike price
        T: Time to maturity
        alpha: Volatility level parameter
        beta: Elasticity parameter (0 <= beta <= 1)
        rho: Correlation between asset and volatility
        nu: Volatility of volatility (vol-of-vol)
    
    Returns:
        Implied volatility from the SABR model
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0 or nu <= 0:
        return 0.01
    
    # Avoid numerical issues when F and K are close
    if abs(F - K) < 1e-6:
        # ATM volatility
        z = nu / alpha * (F ** (1 - beta))
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        vol = alpha / (F ** (1 - beta)) * (z / x)
        return vol
    
    # Compute the log-moneyness
    f_k = np.log(F / K)
    
    # Compute the sigma_ATM (at-the-money volatility)
    z = nu / alpha * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    
    # Handle the case where x is small
    if abs(x) < 1e-6:
        sigma_ATM = alpha / ((F * K) ** ((1 - beta) / 2)) * (1 + ((1 - beta)**2 / 24) * (np.log(F / K))**2 + ((1 - beta)**4 / 1920) * (np.log(F / K))**4)
    else:
        sigma_ATM = alpha / ((F * K) ** ((1 - beta) / 2)) * (z / x)
    
    # Compute the skew term
    skew_term = 1 + ((2 * rho * nu) / alpha) * (F ** ((1 - beta) / 2)) + ((1 - beta)**2 * nu**2) / (12 * alpha**2) * (F * K) ** (1 - beta)
    skew_term *= T
    
    # Compute the final implied volatility
    vol = sigma_ATM * skew_term
    
    # Bound the volatility to reasonable values
    vol = max(vol, 0.01)
    vol = min(vol, 2.0)
    
    return vol

def sabr_price_error(params, F: float, K: float, T: float, market_vol: float) -> float:
    """
    Compute the error between SABR implied volatility and market implied volatility.
    
    Args:
        params: SABR parameters (alpha, beta, rho, nu)
        F: Forward price
        K: Strike price
        T: Time to maturity
        market_vol: Market implied volatility
    
    Returns:
        Squared error
    """
    alpha, beta, rho, nu = params
    
    # Bound parameters to reasonable ranges
    alpha = max(alpha, 0.001)
    beta = max(0, min(beta, 1))
    rho = max(-0.99, min(rho, 0.99))
    nu = max(nu, 0.001)
    
    try:
        model_vol = sabr_implied_volatility(F, K, T, alpha, beta, rho, nu)
        error = model_vol - market_vol
        return error ** 2
    except:
        return 1e10

def calibrate_sabr(df: pd.DataFrame) -> dict:
    """
    Calibrate SABR model parameters to market data using nonlinear optimization.
    
    Args:
        df: DataFrame with implied volatilities, strikes, and maturities
    
    Returns:
        Dictionary with optimal SABR parameters
    """
    print("=" * 70)
    print("SABR MODEL CALIBRATION")
    print("=" * 70)
    
    # Prepare data
    F = 100.0  # Forward price (use current price as approximation)
    
    # Filter data for calibration
    data = df[(df['implied_volatility'] > 0) & (df['implied_volatility'] < 2)]
    
    if len(data) < 10:
        print("Not enough data for calibration. Using synthetic data.")
        return calibrate_sabr_synthetic()
    
    # Initial guess for parameters
    initial_params = [0.2, 0.5, 0.0, 0.3]  # alpha, beta, rho, nu
    
    # Bounds for parameters
    bounds = [(0.001, 1.0), (0, 1), (-0.99, 0.99), (0.001, 1.0)]
    
    print("Calibrating SABR model...")
    print(f"Data points: {len(data)}")
    print(f"Initial parameters: alpha={initial_params[0]:.4f}, beta={initial_params[1]:.4f}, "
          f"rho={initial_params[2]:.4f}, nu={initial_params[3]:.4f}")
    
    # Use differential evolution for global optimization
    def objective(params):
        total_error = 0
        for _, row in data.iterrows():
            K = row['strike']
            T = row['time_to_maturity']
            market_vol = row['implied_volatility']
            total_error += sabr_price_error(params, F, K, T, market_vol)
        return total_error / len(data)
    
    # Differential evolution for global search
    result = differential_evolution(objective, bounds, maxiter=50, popsize=20, disp=False)
    
    # Refine with local optimization
    refined_result = minimize(objective, result.x, method='L-BFGS-B', bounds=bounds)
    
    optimal_params = refined_result.x
    
    print("\nCalibration Results:")
    print(f"Optimal parameters:")
    print(f"  alpha (volatility level): {optimal_params[0]:.4f}")
    print(f"  beta (elasticity):        {optimal_params[1]:.4f}")
    print(f"  rho (correlation):        {optimal_params[2]:.4f}")
    print(f"  nu (vol-of-vol):          {optimal_params[3]:.4f}")
    print(f"  Objective value:          {refined_result.fun:.6f}")
    
    return {
        'alpha': optimal_params[0],
        'beta': optimal_params[1],
        'rho': optimal_params[2],
        'nu': optimal_params[3],
        'F': F
    }

def calibrate_sabr_synthetic() -> dict:
    """
    Calibrate SABR parameters to synthetic data for testing.
    """
    print("\nUsing synthetic data for calibration...")
    
    # Generate synthetic SABR data
    true_params = {'alpha': 0.25, 'beta': 0.5, 'rho': -0.3, 'nu': 0.4}
    F = 100.0
    T = 0.5
    
    strikes = np.linspace(60, 140, 30)
    market_vols = []
    for K in strikes:
        vol = sabr_implied_volatility(F, K, T, true_params['alpha'], true_params['beta'], 
                                      true_params['rho'], true_params['nu'])
        market_vols.append(vol)
    
    # Fit SABR model to the synthetic data
    def objective(params):
        total_error = 0
        for K, market_vol in zip(strikes, market_vols):
            error = sabr_price_error(params, F, K, T, market_vol)
            total_error += error
        return total_error / len(strikes)
    
    bounds = [(0.001, 1.0), (0, 1), (-0.99, 0.99), (0.001, 1.0)]
    result = differential_evolution(objective, bounds, maxiter=50, popsize=20, disp=False)
    refined_result = minimize(objective, result.x, method='L-BFGS-B', bounds=bounds)
    
    optimal_params = refined_result.x
    
    print("\nCalibration Results (Synthetic Data):")
    print(f"True parameters:")
    print(f"  alpha: {true_params['alpha']:.4f}")
    print(f"  beta:  {true_params['beta']:.4f}")
    print(f"  rho:   {true_params['rho']:.4f}")
    print(f"  nu:    {true_params['nu']:.4f}")
    print(f"\nEstimated parameters:")
    print(f"  alpha: {optimal_params[0]:.4f}")
    print(f"  beta:  {optimal_params[1]:.4f}")
    print(f"  rho:   {optimal_params[2]:.4f}")
    print(f"  nu:    {optimal_params[3]:.4f}")
    
    return {
        'alpha': optimal_params[0],
        'beta': optimal_params[1],
        'rho': optimal_params[2],
        'nu': optimal_params[3],
        'F': F
    }

def plot_sabr_calibration(df: pd.DataFrame, params: dict) -> None:
    """
    Plot SABR model fit against market data.
    
    Args:
        df: DataFrame with market implied volatilities
        params: SABR model parameters
    """
    F = params['F']
    alpha = params['alpha']
    beta = params['beta']
    rho = params['rho']
    nu = params['nu']
    
    plt.figure(figsize=(14, 10))
    
    # Group by maturity
    maturities = sorted(df['time_to_maturity'].unique())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(maturities)))
    
    for i, T in enumerate(maturities):
        if i % 2 == 0:  # Plot every other maturity to avoid clutter
            subset = df[df['time_to_maturity'] == T]
            subset = subset.sort_values('strike')
            
            # Market data
            plt.scatter(subset['strike'], subset['implied_volatility'], 
                       label=f'Market T={T:.2f}', color=colors[i], alpha=0.7, s=50)
            
            # SABR model fit
            strikes = np.linspace(subset['strike'].min(), subset['strike'].max(), 50)
            model_vols = []
            for K in strikes:
                vol = sabr_implied_volatility(F, K, T, alpha, beta, rho, nu)
                model_vols.append(vol)
            plt.plot(strikes, model_vols, '--', linewidth=2, color=colors[i], 
                    label=f'SABR T={T:.2f}')
    
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('SABR Model Calibration vs Market Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_synthetic_data() -> pd.DataFrame:
    """
    Generate synthetic market data for testing SABR calibration.
    """
    np.random.seed(42)
    
    data = []
    F = 100.0
    true_params = {'alpha': 0.25, 'beta': 0.5, 'rho': -0.3, 'nu': 0.4}
    maturities = [0.1, 0.3, 0.5, 0.8, 1.2]
    
    for T in maturities:
        strikes = np.linspace(60, 140, 20)
        for K in strikes:
            vol = sabr_implied_volatility(F, K, T, true_params['alpha'], true_params['beta'],
                                          true_params['rho'], true_params['nu'])
            # Add small noise to simulate market data
            vol += 0.02 * np.random.randn()
            vol = max(vol, 0.05)
            data.append({
                'strike': K,
                'time_to_maturity': T,
                'implied_volatility': vol,
                'current_price': F
            })
    
    return pd.DataFrame(data)

def main():
    """
    Run SABR calibration pipeline.
    """
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION - DAY 3: SABR MODEL CALIBRATION")
    print("=" * 70)
    
    # Try to load real data, fallback to synthetic
    try:
        df = pd.read_csv('data/option_data_SPY.csv')
        print(f"Loaded {len(df)} options from data file")
        
        # Check if we have implied volatilities
        if 'implied_volatility' not in df.columns:
            print("Computing implied volatilities...")
            from implied_volatility import compute_implied_volatilities
            df = compute_implied_volatilities(df)
        
        # Clean data
        df = df.dropna(subset=['implied_volatility'])
        df = df[(df['implied_volatility'] > 0.01) & (df['implied_volatility'] < 2)]
        
        if len(df) < 10:
            print("Not enough real data. Using synthetic data.")
            df = generate_synthetic_data()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("No data file found. Generating synthetic data...")
        df = generate_synthetic_data()
    
    # Calibrate SABR model
    params = calibrate_sabr(df)
    
    # Plot calibration results
    print("\n" + "=" * 70)
    print("DISPLAYING SABR CALIBRATION RESULTS")
    print("=" * 70)
    plot_sabr_calibration(df, params)
    
    # Calculate RMSE
    F = params['F']
    alpha = params['alpha']
    beta = params['beta']
    rho = params['rho']
    nu = params['nu']
    
    errors = []
    for _, row in df.iterrows():
        K = row['strike']
        T = row['time_to_maturity']
        market_vol = row['implied_volatility']
        model_vol = sabr_implied_volatility(F, K, T, alpha, beta, rho, nu)
        errors.append((model_vol - market_vol) ** 2)
    
    rmse = np.sqrt(np.mean(errors))
    print(f"\nSABR Model RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("SABR calibration complete.")
    print("Ready for Day 4: Heston Model Calibration.")
    print("=" * 70)

if __name__ == "__main__":
    main()
