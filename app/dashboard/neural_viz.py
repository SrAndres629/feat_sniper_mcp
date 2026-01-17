import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
import json
from datetime import datetime

# CONFIG
API_URL = "http://localhost:8000/sniper/neural-state"
st.set_page_config(
    page_title="FEAT NEURAL MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# STYLING
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4F4F4F;
        text-align: center;
    }
    .big-number { font-size: 2em; font-weight: bold; }
    .label { font-size: 0.8em; color: #A0A0A0; }
</style>
""", unsafe_allow_html=True)

def fetch_brain_state():
    try:
        resp = requests.get(API_URL, timeout=2)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return data
    except Exception as e:
        return None
    return None

# HEADER
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🧠 FEAT Neural Cortex MRI")
with col_h2:
    st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")

# MAIN LOOP
placeholder = st.empty()

while True:
    state = fetch_brain_state()
    
    if not state or not state.get("timestamp"):
        with placeholder.container():
            st.warning("Waiting for Neural Signal... (Is Nexus running?)")
        time.sleep(2)
        continue
        
    probs = state.get("predictions", {})
    pvp = state.get("pvp_context", {})
    immune = state.get("immune_system", {})
    
    with placeholder.container():
        # --- ROW 1: VITAL SIGNS ---
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">SYMBOL</div>
                <div class="big-number">{state.get('symbol', '---')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">PRICE</div>
                <div class="big-number">{state.get('price', 0.0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            uncert = state.get('uncertainty', 0.0)
            color = "#00FF00" if uncert < 0.3 else "#FF0000"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {color};">
                <div class="label">UNCERTAINTY</div>
                <div class="big-number" style="color: {color};">{uncert:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            status = immune.get('status', 'NORMAL')
            color = "#FF0000" if status != 'NORMAL' else "#00FF00"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {color};">
                <div class="label">IMMUNE SYSTEM</div>
                <div class="big-number" style="color: {color};">{status}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- ROW 2: THE BRAIN (Probabilities) ---
        c_brain, c_eyes = st.columns([1, 2])
        
        with c_brain:
            st.subheader("🤖 Cortex Probabilities")
            labels = ['BUY', 'SELL', 'HOLD']
            values = [probs.get('buy', 0), probs.get('sell', 0), probs.get('hold', 0)]
            colors = ['#00CC96', '#EF553B', '#636EFA']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker_colors=colors)])
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=250, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # --- ROW 3: THE EYES (PVP Structure) ---
        with c_eyes:
            st.subheader("👁️ Institutional Vision (PVP)")
            
            # Simple Gauge for Position in Range
            vah = pvp.get('vah', 0)
            val = pvp.get('val', 0)
            poc = pvp.get('poc', 0)
            price = state.get('price', 0)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "number+gauge+delta",
                value = price,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Price vs Structure"},
                delta = {'reference': poc, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [val - 10, vah + 10], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white"},
                    'bgcolor': "black",
                    'steps': [
                        {'range': [val, vah], 'color': "rgba(0, 100, 255, 0.3)"},
                        {'range': [poc-0.5, poc+0.5], 'color': "rgba(255, 0, 0, 0.8)"}],
                    'threshold': {
                        'line': {'color': "yellow", 'width': 4},
                        'thickness': 0.75,
                        'value': price}}))
            
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Dist to POC", f"{pvp.get('dist_poc', 0):.2f} σ")
            m2.metric("In Value Area?", "YES" if pvp.get('pos_in_va') > 0.5 else "NO")
            m3.metric("Energy Score", f"{int(pvp.get('energy', 0))}")

    time.sleep(1) # Refresh Rate
