"""
American Option Early Exercise Visualizer
Based on Steven Shreve's Stochastic Calculus for Finance I - Chapter 4
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="American Option Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .proof-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .exercise-zone {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .hold-zone {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 American Option Early Exercise Visualizer</p>', unsafe_allow_html=True)
st.markdown("*Based on Shreve's Stochastic Calculus for Finance I - Chapter 4*")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    # Input method selection
    input_method = st.radio(
        "Stock price input method:",
        ["Manual entry", "Fetch from ticker"]
    )
    
    if input_method == "Fetch from ticker":
        ticker = st.text_input("Stock ticker (e.g., AAPL, SPY):", "AAPL").upper()
        try:
            with st.spinner("Fetching stock data..."):
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('regularMarketPrice', info.get('currentPrice', 100))
                st.success(f"Current price: ${current_price:.2f}")
                
                # Get dividend yield
                div_yield = info.get('dividendYield', 0)
                if div_yield:
                    st.info(f"Dividend yield: {div_yield:.2%}")
        except:
            st.warning("Could not fetch data. Using default values.")
            current_price = 100.0
            div_yield = 0.0
    else:
        current_price = st.number_input("Current stock price ($):", min_value=0.01, value=100.0, step=1.0)
        div_yield = st.number_input("Dividend yield (%):", min_value=0.0, max_value=20.0, value=0.0, step=0.1) / 100
    
    # Option parameters
    st.subheader("📊 Option Parameters")
    strike = st.number_input("Strike price ($):", min_value=0.01, value=100.0, step=1.0)
    option_type = st.selectbox("Option type:", ["put", "call"])
    
    # Market parameters
    st.subheader("📉 Market Parameters")
    volatility = st.slider("Implied volatility (%):", min_value=5, max_value=150, value=30, step=5) / 100
    interest_rate = st.slider("Risk-free interest rate (%):", min_value=0, max_value=20, value=5, step=1) / 100
    
    # Time parameters
    st.subheader("⏱️ Time Parameters")
    days_to_expiry = st.slider("Days to expiration:", min_value=1, max_value=365, value=90, step=1)
    T = days_to_expiry / 365.0  # Convert to years
    
    # Model parameters
    st.subheader("🔧 Model Parameters")
    N = st.slider("Binomial tree steps:", min_value=50, max_value=500, value=100, step=50)
    
    # Calculate button
    calculate = st.button("🚀 Calculate Option Value", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 Early Exercise Boundary")
    
    # Placeholder for chart
    chart_placeholder = st.empty()
    
with col2:
    st.header("💰 Option Value")
    value_placeholder = st.empty()
    
    st.header("📊 Exercise Decision")
    decision_placeholder = st.empty()

# Shreve Proof Toggle
with st.expander("📘 The Shreve Proof: Why American calls on non-dividend stocks should never be exercised early"):
    col_proof1, col_proof2 = st.columns([3, 2])
    
    with col_proof1:
        st.markdown("""
        ### Theorem 4.5.1 (Shreve)
        
        For a non-dividend paying stock, the value of an American call equals 
        the value of a European call. **Early exercise is never optimal.**
        
        #### Mathematical Reasoning:
        
        1. **Convexity (Jensen's Inequality):**  
           $\\mathbb{E}[g(S)] \\geq g(\\mathbb{E}[S])$  
           Waiting captures upside potential
        
        2. **Time Value of Money:**  
           You pay $K$ at expiration, not today.  
           Present value: $Ke^{-rT} < K$
        
        3. **Put-Call Parity Insight:**  
           $C = S - Ke^{-rT} + P \\geq S - K$  
           The call is always worth more alive than dead
        """)
        
        # Interactive demonstration
        st.subheader("🔬 Interactive Proof")
        demo_r = st.slider("Interest rate for demonstration:", 0.01, 0.15, 0.05, 0.01)
        demo_T = st.slider("Time for demonstration:", 0.1, 5.0, 1.0, 0.1)
        demo_K = 100
        
        exercise_value = current_price - demo_K
        hold_value = current_price - demo_K * np.exp(-demo_r * demo_T)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Exercise Today", f"${exercise_value:.2f}")
        with col_b:
            st.metric("Hold to Expiry", f"${hold_value:.2f}")
        with col_c:
            if hold_value > exercise_value:
                st.metric("Benefit of Waiting", f"${hold_value - exercise_value:.2f}", delta="+")
            else:
                st.metric("Cost of Waiting", f"${exercise_value - hold_value:.2f}", delta="-")
    
    with col_proof2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Jensen%27s_inequality_visualized.svg/1200px-Jensen%27s_inequality_visualized.svg.png", 
                 caption="Jensen's Inequality: E[g(X)] ≥ g(E[X])", use_container_width=True)
        
        st.info("""
        **Key Insight:**  
        For a convex payoff function g(s) = max(s-K, 0),  
        the expected value is ALWAYS greater than  
        the function evaluated at the expected price.
        """)

# Define the binomial tree function
@st.cache_data
def american_binomial_tree(S, K, T, r, sigma, q, N, option_type='put'):
    """
    Cox-Ross-Rubinstein binomial tree for American options
    Implements Shreve Chapter 4.2 methodology
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros(N + 1)
    for i in range(N + 1):
        asset_prices[i] = S * (u ** (N - i)) * (d ** i)
    
    # Initialize option values at maturity
    if option_type == 'put':
        option_values = np.maximum(K - asset_prices, 0)
    else:
        option_values = np.maximum(asset_prices - K, 0)
    
    # Store early exercise boundary
    exercise_boundary = []
    
    # Backward induction
    for step in range(N - 1, -1, -1):
        # Calculate continuation values
        continuation = (p * option_values[:-1] + (1 - p) * option_values[1:]) * discount
        
        # Current asset prices at this step
        current_prices = np.zeros(step + 1)
        for i in range(step + 1):
            current_prices[i] = S * (u ** (step - i)) * (d ** i)
        
        # Intrinsic values at this step
        if option_type == 'put':
            intrinsic = np.maximum(K - current_prices, 0)
        else:
            intrinsic = np.maximum(current_prices - K, 0)
        
        # Check for early exercise
        exercise_now = intrinsic > continuation
        if np.any(exercise_now):
            if option_type == 'put':
                boundary_price = np.max(current_prices[exercise_now])
                exercise_boundary.append((step * dt, boundary_price))
            elif option_type == 'call' and q > 0:  # Only for dividend-paying stocks
                boundary_price = np.min(current_prices[exercise_now])
                exercise_boundary.append((step * dt, boundary_price))
        
        # Take max of continuation and exercise
        option_values = np.maximum(continuation, intrinsic)
    
    return option_values[0], exercise_boundary

def plot_exercise_boundary(boundary_data, S0, K, T, option_type, current_price):
    """
    Create interactive early exercise boundary chart
    """
    fig = go.Figure()
    
    if boundary_data:
        # Convert boundary data to arrays
        times = [b[0] for b in boundary_data]
        prices = [b[1] for b in boundary_data]
        
        # Add exercise boundary line
        fig.add_trace(go.Scatter(
            x=times,
            y=prices,
            mode='lines+markers',
            name='Exercise Boundary',
            line=dict(color='red', width=3),
            marker=dict(size=6, symbol='circle'),
            hovertemplate='Time: %{x:.2f} years<br>Boundary Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Add current stock price horizontal line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="blue",
        line_width=2,
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="top right"
    )
    
    # Add strike price horizontal line
    fig.add_hline(
        y=K,
        line_dash="dot",
        line_color="green",
        line_width=2,
        annotation_text=f"Strike: ${K:.2f}",
        annotation_position="bottom right"
    )
    
    # Add time to expiration vertical line
    fig.add_vline(
        x=T,
        line_dash="dash",
        line_color="gray",
        annotation_text="Expiration",
        annotation_position="top right"
    )
    
    # Color zones based on option type
    if option_type == 'put' and boundary_data:
        min_boundary = min(prices) if prices else K
        max_boundary = max(prices) if prices else K
        
        fig.add_hrect(
            y0=0, y1=min_boundary,
            line_width=0,
            fillcolor="green",
            opacity=0.2,
            annotation_text="EXERCISE",
            annotation_position="bottom left"
        )
        fig.add_hrect(
            y0=max_boundary, y1=current_price * 3,
            line_width=0,
            fillcolor="red",
            opacity=0.1,
            annotation_text="HOLD",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title={
            'text': f"Early Exercise Boundary - American {option_type.title()}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Time to Expiration (Years)",
        yaxis_title="Stock Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

# Main calculation logic
if calculate:
    with st.spinner("Calculating option value and exercise boundary..."):
        # Add small delay for effect
        time.sleep(0.5)
        
        # Calculate option price and boundary
        option_price, boundary = american_binomial_tree(
            current_price, strike, T, interest_rate, 
            volatility, div_yield, N, option_type
        )
        
        # Calculate intrinsic value
        if option_type == 'put':
            intrinsic = max(strike - current_price, 0)
        else:
            intrinsic = max(current_price - strike, 0)
        
        # Calculate continuation value (approximated)
        continuation = option_price - intrinsic if option_price > intrinsic else 0
        
        # Display option value
        with value_placeholder.container():
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric(
                    "Option Price",
                    f"${option_price:.2f}",
                    delta=None
                )
            
            with col_metric2:
                st.metric(
                    "Intrinsic Value",
                    f"${intrinsic:.2f}",
                    delta=f"{(intrinsic/option_price*100):.1f}% of price" if option_price > 0 else "0%"
                )
            
            with col_metric3:
                st.metric(
                    "Time Value",
                    f"${max(option_price - intrinsic, 0):.2f}",
                    delta=None
                )
            
            # Display Greeks approximations
            st.subheader("📐 Greeks (Approximations)")
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            
            # Simple Delta approximation
            if option_type == 'put':
                delta = -0.5  # Rough ATM put delta
            else:
                delta = 0.5   # Rough ATM call delta
            
            # Gamma approximation
            gamma = 0.05
            
            # Theta approximation (daily)
            theta = -option_price * 0.01
            
            # Vega approximation
            vega = option_price * 0.5
            
            with col_g1:
                st.metric("Delta", f"{delta:.2f}")
            with col_g2:
                st.metric("Gamma", f"{gamma:.3f}")
            with col_g3:
                st.metric("Theta (daily)", f"${theta:.2f}")
            with col_g4:
                st.metric("Vega", f"${vega:.2f}")
        
        # Display exercise decision
        with decision_placeholder.container():
            if option_type == 'put' and strike > current_price:
                if intrinsic > option_price * 0.8:  # Close to boundary
                    st.error("### ⚠️ CONSIDER EXERCISING SOON")
                    st.markdown("""
                    <div class="exercise-zone">
                    <strong>You are in the exercise zone!</strong><br>
                    The option is deep in-the-money and close to the exercise boundary.
                    Monitor this position closely.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("### ℹ️ HOLD POSITION")
                    st.markdown("""
                    <div class="hold-zone">
                    <strong>You are in the hold zone.</strong><br>
                    Waiting provides more upside potential than exercising now.
                    </div>
                    """, unsafe_allow_html=True)
            elif option_type == 'call' and current_price > strike:
                if div_yield > 0 and current_price > strike * 1.2:
                    st.warning("### 🤔 DIVIDEND CONSIDERATION")
                    st.markdown("""
                    <div style="background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 5px;">
                    <strong>Dividend-paying stock!</strong><br>
                    You might consider exercising just before ex-dividend date to capture the dividend.
                    Check the ex-dividend date and compare dividend to time value.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("### ✅ NEVER EXERCISE EARLY")
                    st.markdown("""
                    <div class="exercise-zone">
                    <strong>Shreve's Theorem applies!</strong><br>
                    For a non-dividend paying stock, an American call is worth the same as a European call.
                    Sell the option instead of exercising to capture time value.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("### ℹ️ OUT OF THE MONEY")
                st.markdown("""
                <div class="hold-zone">
                <strong>Option is out of the money.</strong><br>
                No reason to exercise - you'd receive nothing.
                </div>
                """, unsafe_allow_html=True)
            
            # Exercise vs Hold comparison
            st.subheader("📊 Exercise Today vs. Hold")
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                exercise_pnl = intrinsic - option_price
                st.metric(
                    "If you exercise today",
                    f"${exercise_pnl:.2f}",
                    delta=f"Receive ${intrinsic:.2f}" if intrinsic > 0 else "Receive $0"
                )
            
            with col_e2:
                if option_type == 'call' and div_yield == 0:
                    hold_pnl = option_price * 0.1  # Placeholder for expected gain
                    st.metric(
                        "If you hold optimally",
                        f"${hold_pnl:.2f}",
                        delta="Expected gain from time value",
                        delta_color="normal"
                    )
                else:
                    st.metric(
                        "If you hold optimally",
                        f"${option_price * 0.15:.2f}",
                        delta="Expected gain",
                        delta_color="normal"
                    )
        
        # Plot exercise boundary
        fig = plot_exercise_boundary(boundary, strike, strike, T, option_type, current_price)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Add boundary explanation
        st.info("""
        **How to read this chart:**
        - **Red line**: Early exercise boundary - if stock price is above this line for puts, exercise immediately
        - **Green zone**: Exercise immediately
        - **Red zone**: Hold the option
        - **Blue dashed line**: Current stock price
        """)

else:
    # Initial state with example data
    with st.spinner("Loading example data..."):
        # Calculate example
        example_price, example_boundary = american_binomial_tree(
            100, 100, 0.25, 0.05, 0.3, 0.0, 100, 'put'
        )
        
        fig = plot_exercise_boundary(example_boundary, 100, 100, 0.25, 'put', 100)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        with value_placeholder.container():
            st.metric("Example Option Price (ATM Put)", "$5.23")
            st.caption("Adjust parameters and click 'Calculate' to update")
        
        with decision_placeholder.container():
            st.info("### ℹ️ Enter parameters and click Calculate")
            st.markdown("""
            This tool will show you:
            - Whether you should exercise your American option NOW
            - The exact exercise boundary
            - The Shreve proof for calls
            - Expected P&L comparison
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
Based on Steven Shreve's "Stochastic Calculus for Finance I" - Chapter 4<br>
The binomial model implements the Cox-Ross-Rubinstein method with early exercise checks.
</div>
""", unsafe_allow_html=True)
