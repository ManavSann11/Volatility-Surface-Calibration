# Volatility Surface Calibration

A Python-based tool for calibrating SABR and Heston stochastic volatility models to observed market option data. The project collects real option chain data via the Yahoo Finance API, calibrates model parameters using nonlinear optimization, and constructs an interactive volatility surface interpolator.

## Project Overview

Volatility surfaces are a fundamental tool in quantitative finance, representing implied volatility as a function of strike price and time to maturity. This project demonstrates the end-to-end process of:

1. Collecting real market data from Yahoo Finance
2. Extracting implied volatilities from option chains
3. Calibrating stochastic volatility models (SABR and Heston)
4. Interpolating and visualizing the volatility surface
5. Computing risk sensitivities for option portfolio management

## Mathematical Background

### Implied Volatility

Implied volatility is the volatility parameter that, when input into an option pricing model (such as Black-Scholes), makes the model price equal to the observed market price. The volatility surface is the relationship between implied volatility, strike price, and time to maturity.

### The Volatility Smile

In practice, implied volatility is not constant across strikes. For equity options, implied volatility tends to be higher for deep out-of-the-money and deep in-the-money options, creating a "smile" or "skew" shape. This phenomenon reflects market participants' expectations of extreme price movements and the leverage effect (volatility increases as prices fall).

### SABR Model

The SABR (Stochastic Alpha Beta Rho) model is a stochastic volatility model that captures the volatility smile observed in equity and interest rate markets. It is widely used for pricing options and managing volatility risk.

The SABR model describes the dynamics of the forward price $F$ and volatility $\alpha$:

$$ dF = \alpha F^\beta dW_1 $$

$$ d\alpha = \nu \alpha dW_2 $$

$$ dW_1 dW_2 = \rho dt $$

Where:
- $\beta$ = elasticity parameter (controls the shape of the volatility smile)
- $\nu$ = volatility of volatility (vol-of-vol)
- $\rho$ = correlation between the asset price and volatility

The SABR model is particularly valued for its ability to produce closed-form approximations for option prices and its intuitive parameters.

### Heston Model

The Heston model is a stochastic volatility model that assumes volatility follows a mean-reverting square-root process:

$$ dS_t = \mu S_t dt + \sqrt{\nu_t} S_t dW_t^S $$

$$ d\nu_t = \kappa (\theta - \nu_t) dt + \sigma \sqrt{\nu_t} dW_t^\nu $$

$$ dW_t^S dW_t^\nu = \rho dt $$

Where:
- $\kappa$ = speed of mean reversion
- $\theta$ = long-term average variance
- $\sigma$ = volatility of variance (vol-of-vol)
- $\rho$ = correlation between asset price and variance

The Heston model offers a realistic term structure of volatility and can capture the skew observed in equity markets through the correlation parameter $\rho$.

### Calibration Objective

Calibration involves finding the model parameters that minimize the difference between model-implied volatilities and market-implied volatilities:

$$ \min_{\Theta} \sum_{i=1}^{N} \left( \sigma_{\text{model}}(K_i, T_i; \Theta) - \sigma_{\text{market}}(K_i, T_i) \right)^2 $$

Where:
- $\Theta$ = model parameters (SABR or Heston)
- $N$ = number of options in the dataset
- $\sigma_{\text{model}}$ = implied volatility from the model
- $\sigma_{\text{market}}$ = implied volatility from market prices

The objective is typically solved using nonlinear least squares optimization, with RMSE (Root Mean Square Error) as a key metric:

$$ \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \sigma_{\text{model},i} - \sigma_{\text{market},i} \right)^2 } $$

### Greeks for Risk Management

The calibrated volatility surface enables the calculation of option price sensitivities (Greeks) for risk management:

- **Delta ($\Delta$)**: Sensitivity of option price to changes in the underlying asset price
- **Gamma ($\Gamma$)**: Sensitivity of delta to changes in the underlying asset price
- **Vega ($\mathcal{V}$)**: Sensitivity of option price to changes in implied volatility

## Project Progression

### Day 1: Data Collection
Collects option chain data for 200+ strike-maturity pairs from Yahoo Finance, extracting strike prices, maturities, and market prices for calls and puts.

### Day 2: Implied Volatility Extraction
Computes implied volatilities from market option prices using the Black-Scholes model and prepares the data for calibration.

### Day 3: SABR Model Calibration
Implements the SABR model and calibrates its parameters ($\beta$, $\nu$, $\rho$) to the observed volatility smile using nonlinear optimization.

### Day 4: Heston Model Calibration
Implements the Heston model and calibrates its parameters ($\kappa$, $\theta$, $\sigma$, $\rho$) to the observed volatility smile.

### Day 5: Model Comparison
Compares the fitted volatility surfaces from both models, analyzing their accuracy and fit quality.

### Day 6: Interactive Volatility Surface
Constructs an interactive volatility surface interpolator that visualizes the surface across strikes and maturities.

### Day 7: Risk Sensitivities
Computes option Greeks ($\Delta$, $\Gamma$, Vega) using the calibrated surface for option risk management.

## Results

The calibrated models achieve:
- **RMSE**: Less than 3% against observed market volatility smiles
- **Model Fit**: Both SABR and Heston models capture the volatility smile effectively
- **Risk Management**: Interactive surface enables visualization of Greek sensitivities

## Repository Structure
volatility-surface-calibration/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_data_collection.py
├── 02_implied_volatility.py
├── 03_sabr_calibration.py
├── 04_heston_calibration.py
├── 05_model_comparison.py
├── 06_volatility_surface.py
├── 07_risk_sensitivities.py

## Tech Stack

- Python 3
- yfinance for market data
- NumPy, SciPy for numerical computation and optimization
- Plotly/Dash for interactive visualizations
- Matplotlib for static plots

## Repository Structure
