import streamlit as st
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# [LEVEL 51] NEURAL CORTEX VISUALIZER
# A "Doctorate Level" interface for observing the Quantum Brain in real-time.

# 1. Page Config (Immersive Mode)
st.set_page_config(
    page_title="FEAT NEXUS | Visual Cortex",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Sci-Fi" Aesthetic
st.markdown("""
    <style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background: #0e1117;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2e3b4e;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    h1, h2, h3 {
        color: #00ffcc;
        font-family: 'Courier New', monospace;
    }
    .stProgress > div > div > div > div {
        background-color: #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Data Connection
STATE_FILE = "data/live_state.json"

def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

# 3. Component Rendering
def render_neural_gauge(prob, direction="BUY"):
    color = "green" if direction == "BUY" else "red"
    if direction == "HOLD": color = "gray"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"NEURAL CONFIDENCE ({direction})", 'font': {'size': 20, 'color': color}},
        delta = {'reference': 80, 'increasing': {'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 50], 'color': '#333'},
                {'range': [50, 80], 'color': '#555'},
                {'range': [80, 100], 'color': '#1e2130'} # Zone
            ],
            'threshold': {
                'line': {'color': "cyan", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Courier New"})
    return fig

def render_tensor_map(kinetic):
    # Radar Chart for Kinetic Tensor
    categories = ['Pattern Strength', 'Coherence', 'Alignment', 'Bias Slope']
    
    # Normalize for visual
    pid = min(kinetic.get("pattern_id", 0) * 20, 100) # Scale 0-4 to 0-100
    coh = kinetic.get("coherence", 0) * 100
    # Alignment is -1 to 1. Map to 0-100.
    # We visualizing intensity, so abs(alignment)
    align = abs(float(kinetic.get("coherence", 0))) * 100 
    
    values = [pid, coh, align, 50] # Placeholder for Bias
    
    fig = px.line_polar(r=values, theta=categories, line_close=True)
    fig.update_traces(fill='toself', line_color='#00ffcc')
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="#1e2130",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor="#333"),
            angularaxis=dict(linecolor="#333", color="white")
        ),
        font=dict(color="white")
    )
    return fig

def render_energy_heatmap(energy_map):
    # energy_map is a list of lists (50x50)
    if not energy_map:
        return None
    
    z_data = np.array(energy_map)
    # Origin lower means low price bins at bottom
    fig = px.imshow(
        z_data,
        labels=dict(x="Time", y="Price Bins", color="Energy"),
        color_continuous_scale='Viridis',
        origin='lower'
    )
    
    fig.update_layout(
        title="SPATIO-TEMPORAL ENERGY MANIFOLD",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e2130",
        font=dict(color="white", family="Courier New"),
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_showscale=False
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    return fig

# 4. Main App Loop
st.title("FEAT NEXUS // VISUAL CORTEX")

# Auto-Refresh Placeholder
placeholder = st.empty()

while True:
    state = load_state()
    
    if state:
        with placeholder.container():
            # Header
            c1, c2, c3, c4 = st.columns(4)
            data = state.get("predictions", {})
            p_buy = data.get("buy", 0.0)
            p_sell = data.get("sell", 0.0)
            
            direction = "HOLD"
            conf = 0.0
            if p_buy > 0.6: direction = "BUY"; conf = p_buy
            elif p_sell > 0.6: direction = "SELL"; conf = p_sell
            
            # [LEVEL 57] Doctoral Metrics Injection
            alpha = state.get("alpha_multiplier", 1.0)
            win_conf = state.get("win_confidence", conf)
            vol = state.get("volatility_regime", 0.0)
            
            c1.metric("SYMBOL", state.get("symbol", "---"))
            c2.metric("WIN CONF.", f"{win_conf*100:.1f}%")
            c3.metric("ALPHA MULTI", f"x{alpha:.2f}")
            c4.metric("VOLATILITY", f"{vol:.2f}")

            # Main Vis
            st.divider()
            m1, m2 = st.columns([2, 1])
            
            with m1:
                st.plotly_chart(render_neural_gauge(win_conf, direction), use_container_width=True)
                
            with m2:
                st.subheader("Kinetic Tensor")
                k_ctx = state.get("kinetic_context", {})
                st.info(f"Pattern: {k_ctx.get('label', 'UNKNOWN')}")
                st.progress(min(k_ctx.get("coherence", 0), 1.0))
                st.caption(f"Coherence: {k_ctx.get('coherence', 0):.2f}")
                
            # PVP Context
            st.divider()
            st.subheader("PVP Context Space")
            pvp = state.get("pvp_context", {})
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("POC Distance", f"{pvp.get('dist_poc', 0):.4f}")
            col_b.metric("In Value Area", "YES" if pvp.get("pos_in_va") > 0 else "NO")
            col_c.metric("Energy Score", f"{pvp.get('energy', 0):.2f}")
            
            # [LEVEL 54] VISUAL CORTEX HEATMAP
            map_data = state.get("spatial_map", [])
            if map_data:
                fig_map = render_energy_heatmap(map_data)
                if fig_map:
                    st.plotly_chart(fig_map, use_container_width=True)
            
            # Heatmap below metrics
            st.divider()
            map_data = state.get("spatial_map", [])
            if map_data:
                import numpy as np # Needed for render helper
                fig_map = render_energy_heatmap(map_data)
                if fig_map:
                    st.plotly_chart(fig_map, use_container_width=True)

    else:
        with placeholder.container():
            st.warning("📡 Waiting for Neural Link... (mcp_server.py not running?)")
            
    time.sleep(1) # Fast Poll
