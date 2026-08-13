"""
Volatility Surface Calibration - Day 7: Risk Sensitivities
Computes and visualizes option Greeks and risk metrics for portfolio management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_greeks(S0, K, T, r, sigma):
    """
    Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho) using Black-Scholes.
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Implied volatility
    
    Returns:
        Dictionary with Delta, Gamma, Vega, Theta, and Rho
    """
    if T <= 0 or sigma <= 0:
        return {
            'delta': 0, 
            'gamma': 0, 
            'vega': 0, 
            'theta': 0, 
            'rho': 0
        }
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta: sensitivity to underlying price
    delta = norm.cdf(d1)
    
    # Gamma: sensitivity of delta to underlying price
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    
    # Vega: sensitivity to volatility (per 1% change)
    vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Theta: sensitivity to time decay (per day)
    theta = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta = theta / 365  # Convert to daily theta
    
    # Rho: sensitivity to interest rate (per 1% change)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def generate_option_portfolio():
    """
    Generate a synthetic option portfolio with various positions.
    
    Returns:
        DataFrame with option positions and Greeks
    """
    S0 = 100.0
    r = 0.05
    
    positions = [
        {'type': 'call', 'strike': 95, 'maturity': 0.5, 'quantity': 10},
        {'type': 'call', 'strike': 100, 'maturity': 0.5, 'quantity': -5},
        {'type': 'put', 'strike': 105, 'maturity': 0.5, 'quantity': 8},
        {'type': 'call', 'strike': 100, 'maturity': 1.0, 'quantity': 15},
        {'type': 'put', 'strike': 90, 'maturity': 1.0, 'quantity': -5},
        {'type': 'call', 'strike': 110, 'maturity': 0.25, 'quantity': 7},
        {'type': 'put', 'strike': 100, 'maturity': 0.25, 'quantity': 3},
        {'type': 'call', 'strike': 105, 'maturity': 1.5, 'quantity': -10},
        {'type': 'put', 'strike': 95, 'maturity': 0.75, 'quantity': 12},
        {'type': 'call', 'strike': 120, 'maturity': 1.5, 'quantity': -3},
    ]
    
    portfolio = []
    
    for pos in positions:
        # Simulate implied volatility for each option
        K = pos['strike']
        T = pos['maturity']
        moneyness = K / S0
        
        # Generate volatility smile
        atm_vol = 0.20 + 0.05 * np.sqrt(T)
        if moneyness < 1:
            sigma = atm_vol + 0.4 * (1 - moneyness)**2 * 0.3
        else:
            sigma = atm_vol + 0.2 * (moneyness - 1)**2 * 0.2
        sigma = max(sigma, 0.05)
        
        # Calculate option price
        if pos['type'] == 'call':
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        
        # Calculate Greeks
        greeks = calculate_greeks(S0, K, T, r, sigma)
        
        portfolio.append({
            'type': pos['type'],
            'strike': K,
            'maturity': T,
            'quantity': pos['quantity'],
            'price': price,
            'sigma': sigma,
            'delta': greeks['delta'] * pos['quantity'],
            'gamma': greeks['gamma'] * pos['quantity'],
            'vega': greeks['vega'] * pos['quantity'],
            'theta': greeks['theta'] * pos['quantity'],
            'rho': rho * pos['quantity']
        })
    
    return pd.DataFrame(portfolio)

def analyze_portfolio_risk(portfolio_df, S0=100.0):
    """
    Analyze portfolio risk metrics including Greeks and stress scenarios.
    
    Args:
        portfolio_df: DataFrame with option positions and Greeks
        S0: Current stock price
    
    Returns:
        Dictionary with portfolio risk metrics
    """
    print("=" * 70)
    print("PORTFOLIO RISK ANALYSIS")
    print("=" * 70)
    
    # Aggregate Greeks
    total_delta = portfolio_df['delta'].sum()
    total_gamma = portfolio_df['gamma'].sum()
    total_vega = portfolio_df['vega'].sum()
    total_theta = portfolio_df['theta'].sum()
    total_rho = portfolio_df['rho'].sum()
    
    print("\nPORTFOLIO GREEKS:")
    print("-" * 50)
    print(f"Delta (Δ):     {total_delta:.2f}")
    print(f"Gamma (Γ):     {total_gamma:.2f}")
    print(f"Vega (V):      {total_vega:.2f}")
    print(f"Theta (Θ):     {total_theta:.4f} (per day)")
    print(f"Rho (ρ):       {total_rho:.2f}")
    
    # Portfolio value
    portfolio_value = (portfolio_df['price'] * portfolio_df['quantity']).sum()
    print(f"\nPortfolio Value: ${portfolio_value:,.2f}")
    
    # Analyze individual positions
    print("\nPOSITION BREAKDOWN:")
    print("-" * 80)
    print(f"{'Type':<6} {'Strike':<8} {'Maturity':<10} {'Qty':<6} {'Delta':<10} {'Gamma':<12} {'Vega':<10}")
    print("-" * 80)
    for idx, row in portfolio_df.iterrows():
        print(f"{row['type']:<6} {row['strike']:<8.0f} {row['maturity']:<10.2f} {row['quantity']:<6.0f} "
              f"{row['delta']:<10.2f} {row['gamma']:<12.4f} {row['vega']:<10.2f}")
    
    return {
        'delta': total_delta,
        'gamma': total_gamma,
        'vega': total_vega,
        'theta': total_theta,
        'rho': total_rho,
        'value': portfolio_value
    }

def compute_stress_scenarios(portfolio_df, S0=100.0):
    """
    Compute portfolio P&L under different stress scenarios.
    
    Args:
        portfolio_df: DataFrame with option positions
        S0: Current stock price
    
    Returns:
        Dictionary with stress scenario results
    """
    print("\n" + "=" * 70)
    print("STRESS SCENARIO ANALYSIS")
    print("=" * 70)
    
    r = 0.05
    scenarios = {
        'Base': {'S0': S0, 'vol_shift': 0},
        'Market Crash (-20%)': {'S0': S0 * 0.8, 'vol_shift': 0.1},
        'Market Rally (+20%)': {'S0': S0 * 1.2, 'vol_shift': -0.05},
        'Vol Spike (+50%)': {'S0': S0, 'vol_shift': 0.2},
        'Vol Crush (-30%)': {'S0': S0, 'vol_shift': -0.1},
        'Crash + Vol Spike': {'S0': S0 * 0.8, 'vol_shift': 0.15},
        'Rally + Vol Crush': {'S0': S0 * 1.2, 'vol_shift': -0.08},
    }
    
    results = []
    
    for name, scenario in scenarios.items():
        S0_new = scenario['S0']
        vol_shift = scenario['vol_shift']
        
        pnl = 0
        for idx, row in portfolio_df.iterrows():
            K = row['strike']
            T = row['maturity']
            sigma = row['sigma'] + vol_shift
            sigma = max(sigma, 0.01)
            qty = row['quantity']
            opt_type = row['type']
            
            # Price under new scenario
            if opt_type == 'call':
                d1 = (np.log(S0_new / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                price_new = S0_new * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                d1 = (np.log(S0_new / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                price_new = K * np.exp(-r * T) * norm.cdf(-d2) - S0_new * norm.cdf(-d1)
            
            pnl += (price_new - row['price']) * qty
        
        results.append({
            'scenario': name,
            'PnL': pnl,
            'S0': S0_new,
            'vol_shift': vol_shift * 100
        })
    
    print("\nSTRESS SCENARIO P&L:")
    print("-" * 60)
    print(f"{'Scenario':<25} {'P&L':<15} {'S0':<10} {'Vol Shift':<12}")
    print("-" * 60)
    for result in results:
        print(f"{result['scenario']:<25} ${result['PnL']:<14,.2f} {result['S0']:<10.0f} {result['vol_shift']:<12.1f}%")
    
    return results

def calculate_value_at_risk(portfolio_df, S0=100.0, n_simulations=10000):
    """
    Calculate Value at Risk using Monte Carlo simulation.
    
    Args:
        portfolio_df: DataFrame with option positions
        S0: Current stock price
        n_simulations: Number of simulations
    
    Returns:
        Dictionary with VaR results
    """
    print("\n" + "=" * 70)
    print("VALUE AT RISK (VaR) ANALYSIS")
    print("=" * 70)
    
    r = 0.05
    sigma_asset = 0.20  # Asset volatility
    T = 0.1  # One month horizon
    
    # Simulate asset returns
    np.random.seed(42)
    returns = np.random.normal(r * T, sigma_asset * np.sqrt(T), n_simulations)
    S0_sim = S0 * np.exp(returns)
    
    # Calculate portfolio value for each simulation
    portfolio_values = []
    
    for S in S0_sim:
        value = 0
        for idx, row in portfolio_df.iterrows():
            K = row['strike']
            T_opt = row['maturity']
            sigma = row['sigma']
            qty = row['quantity']
            opt_type = row['type']
            
            # Adjust time to maturity for remaining time
            T_adj = max(T_opt - T, 0.01)
            sigma_adj = sigma
            
            if T_adj <= 0.01:
                # Option is expiring
                if opt_type == 'call':
                    price = max(S - K, 0)
                else:
                    price = max(K - S, 0)
            else:
                if opt_type == 'call':
                    d1 = (np.log(S / K) + (r + 0.5 * sigma_adj**2) * T_adj) / (sigma_adj * np.sqrt(T_adj))
                    d2 = d1 - sigma_adj * np.sqrt(T_adj)
                    price = S * norm.cdf(d1) - K * np.exp(-r * T_adj) * norm.cdf(d2)
                else:
                    d1 = (np.log(S / K) + (r + 0.5 * sigma_adj**2) * T_adj) / (sigma_adj * np.sqrt(T_adj))
                    d2 = d1 - sigma_adj * np.sqrt(T_adj)
                    price = K * np.exp(-r * T_adj) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            value += price * qty
        
        portfolio_values.append(value)
    
    # Calculate VaR
    pnl = np.array(portfolio_values) - portfolio_df['price'].sum()
    var_95 = np.percentile(pnl, 5)
    var_99 = np.percentile(pnl, 1)
    cvar_95 = pnl[pnl <= var_95].mean()
    cvar_99 = pnl[pnl <= var_99].mean()
    
    print(f"\nValue at Risk (VaR) - 1 Month Horizon:")
    print("-" * 50)
    print(f"95% VaR:  ${var_95:,.2f}")
    print(f"99% VaR:  ${var_99:,.2f}")
    print(f"95% CVaR: ${cvar_95:,.2f}")
    print(f"99% CVaR: ${cvar_99:,.2f}")
    
    # Plot P&L distribution
    plt.figure(figsize=(12, 6))
    plt.hist(pnl, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: ${var_95:,.2f}')
    plt.axvline(var_99, color='orange', linestyle='--', label=f'99% VaR: ${var_99:,.2f}')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.title('Portfolio P&L Distribution (1 Month Horizon)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99
    }

def plot_risk_metrics(portfolio_df):
    """
    Plot risk metrics for the portfolio.
    
    Args:
        portfolio_df: DataFrame with option positions and Greeks
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Delta by strike
    ax = axes[0, 0]
    for opt_type in ['call', 'put']:
        subset = portfolio_df[portfolio_df['type'] == opt_type]
        ax.scatter(subset['strike'], subset['delta'], label=f'{opt_type}', s=50)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Delta')
    ax.set_title('Position Delta by Strike')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Gamma by strike
    ax = axes[0, 1]
    for opt_type in ['call', 'put']:
        subset = portfolio_df[portfolio_df['type'] == opt_type]
        ax.scatter(subset['strike'], subset['gamma'], label=f'{opt_type}', s=50)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Gamma')
    ax.set_title('Position Gamma by Strike')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Vega by maturity
    ax = axes[1, 0]
    for opt_type in ['call', 'put']:
        subset = portfolio_df[portfolio_df['type'] == opt_type]
        ax.scatter(subset['maturity'], subset['vega'], label=f'{opt_type}', s=50)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time to Maturity')
    ax.set_ylabel('Vega')
    ax.set_title('Position Vega by Maturity')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Risk contribution
    ax = axes[1, 1]
    greeks = ['delta', 'gamma', 'vega']
    greek_labels = ['Delta', 'Gamma', 'Vega']
    abs_values = [abs(portfolio_df[g].sum()) for g in greeks]
    ax.bar(greek_labels, abs_values, color='steelblue', alpha=0.7)
    ax.set_ylabel('Absolute Exposure')
    ax.set_title('Total Portfolio Risk Exposure')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Run risk sensitivities analysis pipeline.
    """
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION - DAY 7: RISK SENSITIVITIES")
    print("=" * 70)
    
    # Generate option portfolio
    portfolio_df = generate_option_portfolio()
    print(f"Generated {len(portfolio_df)} option positions")
    
    # Analyze portfolio risk
    risk_metrics = analyze_portfolio_risk(portfolio_df)
    
    # Compute stress scenarios
    stress_results = compute_stress_scenarios(portfolio_df)
    
    # Calculate VaR
    var_results = calculate_value_at_risk(portfolio_df)
    
    # Plot risk metrics
    plot_risk_metrics(portfolio_df)
    
    print("\n" + "=" * 70)
    print("Risk sensitivities analysis complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
