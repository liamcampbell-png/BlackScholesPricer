# Black-Scholes Option Pricing Calculator 

## Overview
An advanced web application built with Streamlit that implements Black-Scholes pricing model and provides interactive displays and visualization

## Features
- **Option Price Calculation**
  - Real-time calculation of call and put option prices
  - Interactive heatmaps showing price sensitivity to stock price and volatility
  - Support for dividend-paying stocks
  - Customizable parameters (stock price, strike price, time to expiration, etc.)

- **Greeks Analysis**
  - Calculation and visualization of all major Greeks:
    - Delta (Δ): Price sensitivity to underlying asset
    - Gamma (Γ): Delta sensitivity to underlying asset
    - Vega (ν): Price sensitivity to volatility
    - Theta (Θ): Price sensitivity to time decay
    - Rho (ρ): Price sensitivity to interest rates
  - Interactive heatmaps for each Greek
  - Detailed mathematical formulas and interpretations

- **Time Decay Analysis**
  - Visual representation of option price decay over time
  - Adjustable time horizon up to 90 days
  - Comparison capability between calls and puts

- **Risk Analysis**
  - Implied volatility calculator using Newton-Rhapson Method
  - Market price comparison tools
  - Real-time risk metrics display

## Technologies Used
- Python 
- Streamlit for web interface
- NumPy for numerical computations
- SciPy for statistical functions
- Matplotlib and Seaborn for visualization
- Pandas for data manipulation

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/black-scholes-pricer.git
cd black-scholes-pricer
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run black_scholes_pricer.py
```

## Usage
1. Input parameters in the sidebar:
   - Stock price (S)
   - Strike price (K)
   - Time to expiration (t)
   - Risk-free rate (r)
   - Dividend yield (q)
   - Volatility (σ)

2. Select option type (Call/Put)
3. Navigate through different tabs to access various analyses
4. Interact with visualizations and calculators

## Mathematical Background
The application implements the Black-Scholes option pricing model, which assumes:
- Log-normal distribution of stock prices
- No arbitrage opportunities
- European-style options
- Constant risk-free rate and volatility
- Efficient markets

