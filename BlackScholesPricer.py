import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#function to calculate blackscholesCall price
def blackScholesCall(S, K, sigma, r, q, t):
    d_1Call = (math.log(S/K) + t * (r - q + 0.5 * (sigma ** 2))) / (sigma * (t ** 0.5))
    d_2Call = d_1Call - (sigma * (t ** 0.5))
    N1Call = norm.cdf(d_1Call)
    N2Call = norm.cdf(d_2Call)
    C = (S * math.exp(-q * t) * N1Call) - (K * math.exp(-r * t) * N2Call)
    return C

#function to calculate put price
def blackScholesPut(S, K, sigma, r, q, t):
    d_1Put = (math.log(S/K) + t * (r - q + 0.5 * (sigma ** 2))) / (sigma * (t ** 0.5))
    d_2Put = d_1Put - (sigma * (t ** 0.5))
    N1Put = norm.cdf(-d_1Put)
    N2Put = norm.cdf(-d_2Put)
    P = (K * math.exp(-r * t) * N2Put) - (S * math.exp(-q * t) * N1Put)
    return P
#function to calculate delta greek
def delta(S, K, sigma, r, q, t, option_type='call'):
    d_1 = (math.log(S/K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * (math.sqrt(t)))
    expMinusQt = math.exp(-q * t)
    if option_type == 'call':
        return expMinusQt * norm.cdf(d_1)
    return expMinusQt * (norm.cdf(d_1) - 1)  

#function to calculate gamma greek
def gamma(S, K, sigma, r, q, t):
    d_1 = (math.log(S/K) + t * (r - q + 0.5 * sigma ** 2)) / (sigma * (t ** 0.5))
    return norm.pdf(d_1) * math.exp(-q * t) / (S * sigma * math.sqrt(t))


#function to calculate vega greek
def vega(S, K, sigma, r, q, t):
    d_1 = (math.log(S/K) + t * (r - q + 0.5 * sigma ** 2)) / (sigma * (t ** 0.5))
    return S * math.sqrt(t) * norm.pdf(d_1) * math.exp(-q * t)

#function to calculate theta greek
def theta(S, K, sigma, r, q, t, option_type='call'):
    d_1 = (math.log(S/K) + t * (r - q + 0.5 * sigma ** 2)) / (sigma * (t ** 0.5))
    d_2 = d_1 - sigma * math.sqrt(t)
    if option_type == 'call':
        theta = -S * norm.pdf(d_1) * sigma * math.exp(-q * t) / (2 * math.sqrt(t)) \
                + q * S * norm.cdf(d_1) * math.exp(-q * t) \
                - r * K * math.exp(-r * t) * norm.cdf(d_2)
    else:
        theta = -S * norm.pdf(d_1) * sigma * math.exp(-q * t) / (2 * math.sqrt(t)) \
                - q * S * norm.cdf(-d_1) * math.exp(-q * t) \
                + r * K * math.exp(-r * t) * norm.cdf(-d_2)
    return theta

#function to calculate rho greek
def rho(S, K, sigma, r, q, t, option_type='call'):
    d_1 = (math.log(S/K) + t * (r - q + 0.5 * sigma ** 2)) / (sigma * (t ** 0.5))
    d_2 = d_1 - sigma * math.sqrt(t)
    if option_type == 'call':
        return K * t * math.exp(-r * t) * norm.cdf(d_2)
    return -K * t * math.exp(-r * t) * norm.cdf(-d_2)

def vega_for_iv(S, K, sigma, r, q, t, option_type='call'):
    d1 = (math.log(S/K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    return S * math.sqrt(t) * norm.pdf(d1) * math.exp(-q * t)

#function to calculate implied volatility using Newton's method
def implied_volatility(market_price, S, K, r, q, t, option_type='call', max_iter=100, precision=1e-5):
   
    #define x_0 using brenner-subrahamanyam approximation
    moneyness = S / K
    if 0.8 <= moneyness <= 1.2:
        sigma = math.sqrt((2*math.pi) / t) * (market_price / S)
    else:
        sigma = 0.2

    for i in range(max_iter):
        if option_type == 'call':
            price = blackScholesCall(S, K, sigma, r, q, t)
        else:
            price = blackScholesPut(S, K, sigma, r, q, t)

        diff = price - market_price
        if abs(diff) < precision:
            return sigma

        v = vega_for_iv(S, K, sigma, r, q, t, option_type)
        if abs(v) < 1e-10:  # Avoid division by zero
            sigma = sigma * 1.5
            continue
        
        # Newton-Raphson update
        sigma = sigma - diff / v
        
        # Keep sigma within reasonable bounds
        if sigma <= 0.0001:
            sigma = 0.0001
        elif sigma > 5:
            sigma = 5.0
    
    return None
    
def calculate_risk_metrics(S, K, sigma, r, q, t, option_type='call'):
    return {
        "Delta": delta(S, K, sigma, r, q, t, option_type),
        "Gamma": gamma(S, K, sigma, r, q, t),
        "Vega": vega(S, K, sigma, r, q, t),
        "Theta": theta(S, K, sigma, r, q, t, option_type),
        "Rho": rho(S, K, sigma, r, q, t, option_type)
    }

#code to plot the heatmap
def plot_heatmap(K, t, r, q, option_type='call', colormap='viridis'):
    stock_prices = np.linspace(K * 0.5, K * 1.5, 100)
    volatilities = np.linspace(0.1, 0.5, 100)
    S_mesh, sigma_mesh = np.meshgrid(stock_prices, volatilities)
    
    prices = np.zeros_like(S_mesh)
    for i in range(len(volatilities)):
        for j in range(len(stock_prices)):
            if option_type == 'call':
                prices[i,j] = blackScholesCall(S_mesh[i,j], K, sigma_mesh[i,j], r, q, t)
            else:
                prices[i,j] = blackScholesPut(S_mesh[i,j], K, sigma_mesh[i,j], r, q, t)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        prices,
        xticklabels=20,
        yticklabels=20,
        cmap=colormap,
        cbar_kws={'label': f'{option_type.capitalize()} Option Price ($)'},
        annot=False
    )
    
    plt.title(f'{option_type.capitalize()} Option Price Heatmap\nStrike Price (K)=${K}, Time to Expiration (t)={t}yr, Risk-Free Interest Rate (r)={r*100}%, Dividend Yield (q)={q*100}%', pad=20)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Volatility (σ)')
    
    x_positions = np.linspace(0, len(stock_prices)-1, 10)
    y_positions = np.linspace(0, len(volatilities)-1, 10)
    plt.xticks(x_positions, np.round(stock_prices[::10], 1), rotation=45)
    plt.yticks(y_positions, np.round(volatilities[::10], 2))
    
    plt.figtext(0.02, -0.1, 
                'Brighter colors indicate higher option prices.\nStock price increases →\nVolatility increases ↑', 
                fontsize=8)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

#code to plot the heatmap for the greeks
def plot_greeks_heatmap(K, t, r, q, greek_type='delta', option_type='call'):
    stock_prices = np.linspace(K * 0.5, K * 1.5, 100)
    volatilities = np.linspace(0.1, 0.5, 100)
    S_mesh, sigma_mesh = np.meshgrid(stock_prices, volatilities)
    
    greek_values = np.zeros_like(S_mesh)
    for i in range(len(volatilities)):
        for j in range(len(stock_prices)):
            if greek_type == 'delta':
                greek_values[i,j] = delta(S_mesh[i,j], K, sigma_mesh[i,j], r, q, t, option_type)
            elif greek_type == 'gamma':
                greek_values[i,j] = gamma(S_mesh[i,j], K, sigma_mesh[i,j], r, q, t)
            elif greek_type == 'vega':
                greek_values[i,j] = vega(S_mesh[i,j], K, sigma_mesh[i,j], r, q, t)
            elif greek_type == 'theta':
                greek_values[i,j] = theta(S_mesh[i,j], K, sigma_mesh[i,j], r, q, t, option_type)
            elif greek_type == 'rho':
                greek_values[i,j] = rho(S_mesh[i,j], K, sigma_mesh[i,j], r, q, t, option_type)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(greek_values, xticklabels=20, yticklabels=20,
                cmap='coolwarm', cbar_kws={'label': f'{greek_type.capitalize()} Value'})
    plt.title(f'{option_type.capitalize()} Option {greek_type.capitalize()} Heatmap\nK=${K}, t={t}yr, r={r*100}%, q={q*100}%')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Volatility (σ)')
    st.pyplot(plt)
    plt.close()

#map to plot put/call option time decay
def plot_time_decay(S, K, sigma, r, q, days=30, option_type='call'):
    times = np.linspace(1/365, days/365, days)
    if option_type == 'call':
        prices = [blackScholesCall(S, K, sigma, r, q, t) for t in times]
    else:
        prices = [blackScholesPut(S, K, sigma, r, q, t) for t in times]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, days+1), prices)
    plt.title(f'{option_type.capitalize()} Option Time Decay')
    plt.xlabel('Days to Expiration')
    plt.ylabel('Option Price ($)')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

#code for main UI
def main():
    # Add LinkedIn link
    st.markdown("""
    <div style="text-align: right">
        <a href="https://www.linkedin.com/in/liam-campbell0/" target="_blank">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg" 
                 alt="LinkedIn" 
                 width="80"/>
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.title("Advanced Black-Scholes Option Pricer")
    st.write("This app calculates option prices and Greeks using the Black-Scholes model.")

    # sidebar for inputs
    st.sidebar.title("Parameters")
    
    # basic parameters
    S = st.sidebar.number_input('Stock Price (S)', min_value=1.0, value=100.0)
    K = st.sidebar.number_input('Strike Price (K)', min_value=1.0, value=100.0)
    
    # time and rate parameters
    t = st.sidebar.slider('Time to Expiration (years)', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    r = st.sidebar.slider('Risk-Free Rate (r)', min_value=0.0, max_value=0.2, value=0.05, step=0.001)
    q = st.sidebar.slider('Dividend Yield (q)', min_value=0.0, max_value=0.2, value=0.03, step=0.001)
    sigma = st.sidebar.slider('Volatility (σ)', min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    
    # option type selection
    option_type = st.sidebar.radio("Option Type:", ["call", "put"])

    # create tabs for different analyses
    tabs = st.tabs(["Pricing", "Greeks", "Time Decay", "Risk Analysis"])
    
    with tabs[0]:
        st.subheader("Option Pricing")
        st.write("### Black-Scholes Option Pricing Formulas")
        st.markdown(r"""
        **Call Option Price (C):**
        
        $C = Se^{-qt}\mathcal{N}(d_1) - Ke^{-rt}\mathcal{N}(d_2)$
        
        **Put Option Price (P):**
        
        $P = Ke^{-rt}\mathcal{N}(-d_2) - Se^{-qt}\mathcal{N}(-d_1)$
        
        where:
        
        $d_1 = \frac{\ln(S/K) + (r-q+\frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$
        
        $d_2 = d_1 - \sigma\sqrt{t} = \frac{\ln(S/K) + (r-q-\frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$
        
        Parameters:
        - S: Current stock price
        - K: Strike price
        - r: Risk-free interest rate
        - q: Dividend yield
        - σ: Volatility
        - t: Time to expiration
        - $\mathcal{N}(x)$: Cumulative standard normal distribution function
        """)
        st.markdown("---")

        if option_type == "call":
            price = blackScholesCall(S, K, sigma, r, q, t)
        else:
            price = blackScholesPut(S, K, sigma, r, q, t)
        
        st.metric(f"{option_type.capitalize()} Option Price", f"${price:.2f}")
    

        st.subheader("Price Heatmaps")
        heatmap_options = st.multiselect(
            "Select heatmaps to display:",
            ["Call Option Price", "Put Option Price"],
            default=["Call Option Price"] if option_type == "call" else ["Put Option Price"]
        )
        
        colormap = st.selectbox("Choose colormap:", 
            ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu'])
        
        # Display selected heatmaps
        if "Call Option Price" in heatmap_options:
            st.write("### Call Option Price Heatmap")
            plot_heatmap(K, t, r, q, "call", colormap)
            
        if "Put Option Price" in heatmap_options:
            st.write("### Put Option Price Heatmap")
            plot_heatmap(K, t, r, q, "put", colormap)
    #tab for greeks 
    with tabs[1]:
        st.subheader("Greeks Analysis")
        greek_type = st.selectbox("Greek to visualize:", ["Delta", "Gamma", "Vega", "Theta", "Rho"])

        greek_info = {
            "Delta": {
                "description": """
                Delta (Δ) measures the rate of change in the option price with respect to the change in the underlying asset's price.
                - For call options: 0 to +1 (higher when in-the-money)
                - For put options: -1 to 0 (higher when in-the-money)
                - Also represents the equivalent stock position for hedging
                """,
                "formula": r"""
                Call Delta = $e^{-qt}N(d_1)$
                
                Put Delta = $e^{-qt}(N(d_1) - 1)$
                
                where:
                $d_1 = \frac{\ln(S/K) + (r-q+\frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$               
                """
            },
            "Gamma": {
                "description": """
                Gamma (Γ) measures the rate of change in Delta with respect to the change in the underlying asset's price.
                - Same for both calls and puts
                - Highest for at-the-money options
                - Represents the curvature of the option's value
                - Key for dynamic hedging strategies
                """,
                "formula": r"""
                Gamma = $\frac{e^{-qt}N'(d_1)}{S\sigma\sqrt{t}}$
                
                where $N'(d_1)$ is the standard normal probability density function
                
                $d_1 = \frac{\ln(S/K) + (r-q+\frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$
                """
            },
            "Vega": {
                "description": """
                Vega (ν) measures the rate of change in the option price with respect to the change in volatility.
                - Same for both calls and puts
                - Highest for at-the-money options
                - Important for volatility trading strategies
                - Expressed as change per 1% change in volatility
                """,
                "formula": r"""
                Vega = $S\sqrt{t}e^{-qt}N'(d_1)$
                
                where $N'(d_1)$ is the standard normal probability density function
                
                $d_1 = \frac{\ln(S/K) + (r-q+\frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$
                """
            },
            "Theta": {
                "description": """
                Theta (Θ) measures the rate of change in the option price with respect to time (time decay).
                - Usually negative (options lose value over time)
                - More negative for at-the-money options
                - Accelerates as expiration approaches
                - Expressed as value change per day
                """,
                "formula": r"""
                Call Theta = $-\frac{Se^{-qt}N'(d_1)\sigma}{2\sqrt{t}} + qSe^{-qt}N(d_1) - rKe^{-rt}N(d_2)$
                
                Put Theta = $-\frac{Se^{-qt}N'(d_1)\sigma}{2\sqrt{t}} - qSe^{-qt}N(-d_1) + rKe^{-rt}N(-d_2)$
                
                where:
                
                $d_1 = \frac{\ln(S/K) + (r-q+\frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$
                
                $d_2 = d_1 - \sigma\sqrt{t}$
                """
            },
            "Rho": {
                "description": """
                Rho (ρ) measures the rate of change in the option price with respect to the risk-free interest rate.
                - Usually positive for calls, negative for puts
                - Larger for in-the-money options
                - More significant for longer-term options
                - Expressed as change per 1% change in rates
                """,
                "formula": r"""
                Call Rho = $Kte^{-rt}N(d_2)$
                
                Put Rho = $-Kte^{-rt}N(-d_2)$
                
                where:
                
                $d_1 = \frac{\ln(S/K) + (r-q+\frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$
                
                $d_2 = d_1 - \sigma\sqrt{t}$
                """
            }
        }
        # display description and formula for selected Greek
        st.write("### Description")
        st.markdown(greek_info[greek_type]["description"])        
        st.write("### Formula")
        st.markdown(greek_info[greek_type]["formula"])

        plot_greeks_heatmap(K, t, r, q, greek_type.lower(), option_type)
        
        # display current Greeks values
        st.write("### Current Values")
        metrics = calculate_risk_metrics(S, K, sigma, r, q, t, option_type)
        cols = st.columns(5)
        for i, (name, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(name, f"{value:.4f}")

    #tab to display time decay analyis
    with tabs[2]:
        st.subheader("Time Decay Analysis")
        days = st.slider("Days to analyze:", 1, 90, 30)
        plot_time_decay(S, K, sigma, r, q, days, option_type)
    
    #tab to display implied volatility
    with tabs[3]:
        st.subheader("Risk Analysis")
        # Add implied volatility calculator
        st.write("Implied Volatility Calculator")
        market_price = st.number_input("Market Price", min_value=0.01, value=price)
        implied_vol = implied_volatility(market_price, S, K, r, q, t, option_type)
        if implied_vol:
            st.metric("Implied Volatility", f"{implied_vol:.2%}")
        else:
            st.error("Could not calculate implied volatility for these parameters")

if __name__ == "__main__":
    main()
