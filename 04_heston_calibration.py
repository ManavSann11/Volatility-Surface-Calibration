"""
Volatility Surface Calibration - Day 4: Heston Model Calibration
Implements the Heston stochastic volatility model and calibrates it to market data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def heston_characteristic_function(phi, S0, K, T, r, kappa, theta, sigma, rho, v0):
    """
    Compute the Heston characteristic function for option pricing using the
    Lewis (2000) formulation.
    
    Args:
        phi: Integration variable
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        kappa: Speed of mean reversion
        theta: Long-term variance
        sigma: Volatility of variance (vol-of-vol)
        rho: Correlation between asset and variance
        v0: Initial variance
    
    Returns:
        Characteristic function value
    """
    # Heston parameters
    a = kappa * theta
    b = kappa
    
    # Complex integration
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 + sigma**2 * (phi * 1j + phi**2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)
    
    C = (r * phi * 1j * T + (a / sigma**2) * ((b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
    D = (b - rho * sigma * phi * 1j + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    
    return np.exp(C + D * v0 + 1j * phi * np.log(S0))

def heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0):
    """
    Compute Heston call option price using the Lewis (2000) integration method.
    
    Args:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        kappa: Speed of mean reversion
        theta: Long-term variance
        sigma: Volatility of variance (vol-of-vol)
        rho: Correlation between asset and variance
        v0: Initial variance
    
    Returns:
        Call option price
    """
    if T <= 0:
        return max(S0 - K, 0)
    
    # Integration limit
    limit = 100
    
    def integrand(phi):
        psi = heston_characteristic_function(phi - 1j*0.5, S0, K, T, r, kappa, theta, sigma, rho, v0)
        psi = psi * np.exp(-1j * phi * np.log(K) + 0.5 * 1j * phi * (r * T))
        return np.real(psi / (phi**2 + 0.25))
    
    # Numerical integration
    try:
        integral, _ = quad(integrand, 0, limit, limit=100)
        price = S0 - np.sqrt(S0 * K) * np.exp(-r * T / 2) / np.pi * integral
        return max(price, 0.001)
    except:
        return max(S0 - K, 0)  # Fallback to intrinsic value

def heston_implied_volatility(S0, K, T, r, kappa, theta, sigma, rho, v0):
    """
    Compute implied volatility from Heston model using Black-Scholes inverse.
    
    Args:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        kappa: Speed of mean reversion
        theta: Long-term variance
        sigma: Volatility of variance (vol-of-vol)
        rho: Correlation between asset and variance
        v0: Initial variance
    
    Returns:
        Implied volatility
    """
    try:
        price = heston_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0)
        
        # Inverse Black-Scholes to get implied volatility
        def objective(vol):
            d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return bs_price - price
        
        # Bracket search for implied volatility
        for vol in np.linspace(0.01, 2.0, 100):
            if objective(vol) > 0:
                break
        
        # Use Brent's method to find the root
        from scipy.optimize import brentq
        iv = brentq(objective, 0.01, 2.0, maxiter=100)
        return iv
    except:
        return 0.25  # Return reasonable default

def heston_price_error(params, S0, K, T, market_vol):
    """
    Compute error between Heston model and market implied volatility.
    
    Args:
        params: Heston parameters (kappa, theta, sigma, rho, v0)
        S0: Initial stock price
        K: Strike price
        T: Time to maturity
        market_vol: Market implied volatility
    
    Returns:
        Squared error
    """
    kappa, theta, sigma, rho, v0 = params
    
    # Bound parameters to reasonable ranges
    kappa = max(kappa, 0.01)
    theta = max(theta, 0.001)
    sigma = max(sigma, 0.01)
    rho = max(-0.99, min(rho, 0.99))
    v0 = max(v0, 0.001)
    
    r = 0.05  # Risk-free rate
    
    try:
        model_vol = heston_implied_volatility(S0, K, T, r, kappa, theta, sigma, rho, v0)
        error = model_vol - market_vol
        return error ** 2
    except:
        return 1e10

def calibrate_heston(df):
    """
    Calibrate Heston model parameters to market data using nonlinear optimization.
    
    Args:
        df: DataFrame with implied volatilities, strikes, and maturities
    
    Returns:
        Dictionary with optimal Heston parameters
    """
    print("=" * 70)
    print("HESTON MODEL CALIBRATION")
    print("=" * 70)
    
    # Prepare data
    S0 = 100.0  # Current price
    
    # Filter data for calibration
    data = df[(df['implied_volatility'] > 0) & (df['implied_volatility'] < 2)]
    
    if len(data) < 10:
        print("Not enough data for calibration. Using synthetic data.")
        return calibrate_heston_synthetic()
    
    # Initial guess for parameters
    initial_params = [2.0, 0.04, 0.3, -0.5, 0.04]  # kappa, theta, sigma, rho, v0
    
    # Bounds for parameters
    bounds = [(0.01, 10.0), (0.001, 1.0), (0.01, 1.0), (-0.99, 0.99), (0.001, 0.5)]
    
    print("Calibrating Heston model...")
    print(f"Data points: {len(data)}")
    print(f"Initial parameters: kappa={initial_params[0]:.4f}, theta={initial_params[1]:.4f}, "
          f"sigma={initial_params[2]:.4f}, rho={initial_params[3]:.4f}, v0={initial_params[4]:.4f}")
    
    # Use differential evolution for global optimization
    def objective(params):
        total_error = 0
        count = 0
        for idx, row in data.iterrows():
            K = row['strike']
            T = row['time_to_maturity']
            market_vol = row['implied_volatility']
            if market_vol > 0:
                error = heston_price_error(params, S0, K, T, market_vol)
                total_error += error
                count += 1
        return total_error / count if count > 0 else 1e10
    
    # Differential evolution for global search
    result = differential_evolution(objective, bounds, maxiter=30, popsize=15, disp=False)
    
    # Refine with local optimization
    refined_result = minimize(objective, result.x, method='L-BFGS-B', bounds=bounds)
    
    optimal_params = refined_result.x
    
    print("\nCalibration Results:")
    print(f"Optimal parameters:")
    print(f"  kappa (mean reversion speed): {optimal_params[0]:.4f}")
    print(f"  theta (long-term variance):   {optimal_params[1]:.4f}")
    print(f"  sigma (vol-of-vol):           {optimal_params[2]:.4f}")
    print(f"  rho (correlation):            {optimal_params[3]:.4f}")
    print(f"  v0 (initial variance):        {optimal_params[4]:.4f}")
    print(f"  Objective value:              {refined_result.fun:.6f}")
    
    return {
        'kappa': optimal_params[0],
        'theta': optimal_params[1],
        'sigma': optimal_params[2],
        'rho': optimal_params[3],
        'v0': optimal_params[4],
        'S0': S0,
        'r': 0.05
    }

def calibrate_heston_synthetic():
    """
    Calibrate Heston parameters to synthetic data for testing.
    """
    print("\nUsing synthetic data for calibration...")
    
    # True Heston parameters
    true_params = {'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.5, 'v0': 0.04}
    S0 = 100.0
    T = 0.5
    r = 0.05
    
    # Generate synthetic data
    strikes = np.linspace(60, 140, 20)
    market_vols = []
    for K in strikes:
        vol = heston_implied_volatility(S0, K, T, r, true_params['kappa'], true_params['theta'],
                                        true_params['sigma'], true_params['rho'], true_params['v0'])
        market_vols.append(vol)
    
    # Fit Heston model to synthetic data
    def objective(params):
        total_error = 0
        for K, market_vol in zip(strikes, market_vols):
            error = heston_price_error(params, S0, K, T, market_vol)
            total_error += error
        return total_error / len(strikes)
    
    bounds = [(0.01, 10.0), (0.001, 1.0), (0.01, 1.0), (-0.99, 0.99), (0.001, 0.5)]
    result = differential_evolution(objective, bounds, maxiter=30, popsize=15, disp=False)
    refined_result = minimize(objective, result.x, method='L-BFGS-B', bounds=bounds)
    
    optimal_params = refined_result.x
    
    print("\nCalibration Results (Synthetic Data):")
    print(f"True parameters:")
    print(f"  kappa: {true_params['kappa']:.4f}")
    print(f"  theta: {true_params['theta']:.4f}")
    print(f"  sigma: {true_params['sigma']:.4f}")
    print(f"  rho:   {true_params['rho']:.4f}")
    print(f"  v0:    {true_params['v0']:.4f}")
    print(f"\nEstimated parameters:")
    print(f"  kappa: {optimal_params[0]:.4f}")
    print(f"  theta: {optimal_params[1]:.4f}")
    print(f"  sigma: {optimal_params[2]:.4f}")
    print(f"  rho:   {optimal_params[3]:.4f}")
    print(f"  v0:    {optimal_params[4]:.4f}")
    
    return {
        'kappa': optimal_params[0],
        'theta': optimal_params[1],
        'sigma': optimal_params[2],
        'rho': optimal_params[3],
        'v0': optimal_params[4],
        'S0': S0,
        'r': r
    }

def plot_heston_calibration(df, params):
    """
    Plot Heston model fit against market data.
    
    Args:
        df: DataFrame with market implied volatilities
        params: Heston model parameters
    """
    S0 = params['S0']
    r = params['r']
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    plt.figure(figsize=(14, 10))
    
    # Group by maturity
    maturities = sorted(df['time_to_maturity'].unique())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(maturities)))
    
    for i, T in enumerate(maturities):
        if i % 2 == 0:
            subset = df[df['time_to_maturity'] == T]
            subset = subset.sort_values('strike')
            
            # Market data
            plt.scatter(subset['strike'], subset['implied_volatility'], 
                       label=f'Market T={T:.2f}', color=colors[i], alpha=0.7, s=50)
            
            # Heston model fit
            strikes = np.linspace(subset['strike'].min(), subset['strike'].max(), 30)
            model_vols = []
            for K in strikes:
                vol = heston_implied_volatility(S0, K, T, r, kappa, theta, sigma, rho, v0)
                model_vols.append(vol)
            plt.plot(strikes, model_vols, '--', linewidth=2, color=colors[i],
                    label=f'Heston T={T:.2f}')
    
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('Heston Model Calibration vs Market Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Run Heston calibration pipeline.
    """
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION - DAY 4: HESTON MODEL CALIBRATION")
    print("=" * 70)
    
    # Try to load real data, fallback to synthetic
    try:
        df = pd.read_csv('data/option_data_SPY.csv')
        print(f"Loaded {len(df)} options from data file")
        
        if 'implied_volatility' not in df.columns:
            print("Computing implied volatilities...")
            from implied_volatility import compute_implied_volatilities
            df = compute_implied_volatilities(df)
        
        df = df.dropna(subset=['implied_volatility'])
        df = df[(df['implied_volatility'] > 0.01) & (df['implied_volatility'] < 2)]
        
        if len(df) < 10:
            print("Not enough real data. Using synthetic data.")
            df = calibrate_heston_synthetic()  # This will generate data
            # Need to recreate a DataFrame for plotting
            data = []
            S0 = 100.0
            true_params = {'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.5, 'v0': 0.04}
            for T in [0.1, 0.3, 0.5, 0.8, 1.2]:
                strikes = np.linspace(60, 140, 20)
                for K in strikes:
                    vol = heston_implied_volatility(S0, K, T, 0.05, 
                                                    true_params['kappa'], true_params['theta'],
                                                    true_params['sigma'], true_params['rho'], 
                                                    true_params['v0'])
                    data.append({'strike': K, 'time_to_maturity': T, 'implied_volatility': vol})
            df = pd.DataFrame(data)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("No data file found. Generating synthetic data...")
        data = []
        S0 = 100.0
        true_params = {'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.5, 'v0': 0.04}
        for T in [0.1, 0.3, 0.5, 0.8, 1.2]:
            strikes = np.linspace(60, 140, 20)
            for K in strikes:
                vol = heston_implied_volatility(S0, K, T, 0.05, 
                                                true_params['kappa'], true_params['theta'],
                                                true_params['sigma'], true_params['rho'], 
                                                true_params['v0'])
                data.append({'strike': K, 'time_to_maturity': T, 'implied_volatility': vol})
        df = pd.DataFrame(data)
    
    # Calibrate Heston model
    params = calibrate_heston(df)
    
    # Plot calibration results
    print("\n" + "=" * 70)
    print("DISPLAYING HESTON CALIBRATION RESULTS")
    print("=" * 70)
    plot_heston_calibration(df, params)
    
    # Calculate RMSE
    S0 = params['S0']
    r = params['r']
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    errors = []
    for idx, row in df.iterrows():
        K = row['strike']
        T = row['time_to_maturity']
        market_vol = row['implied_volatility']
        model_vol = heston_implied_volatility(S0, K, T, r, kappa, theta, sigma, rho, v0)
        errors.append((model_vol - market_vol) ** 2)
    
    rmse = np.sqrt(np.mean(errors))
    print(f"\nHeston Model RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Heston calibration complete.")
    print("Ready for Day 5: Model Comparison.")
    print("=" * 70)

if __name__ == "__main__":
    main()
