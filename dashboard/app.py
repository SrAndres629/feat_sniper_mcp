import streamlit as st
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# [LEVEL 64] C2 COMMAND & CONTROL DASHBOARD
# Architecture: Isolated UI -> Shared State -> Core Node

st.set_page_config(
    page_title="FEAT NEXUS | War Room",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. PREMIUM AESTHETICS (Warfare Grade) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        background-color: #05070a;
    }
    .main {
        background-color: #05070a;
        color: #e0e6ed;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 2px;
        border: 1px solid #2e3b4e;
        background-color: #1a1f26;
        color: #00ffcc;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #00ffcc;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.2);
    }
    .panic-button>button {
        background-color: #441111 !important;
        color: #ff4444 !important;
        border-color: #662222 !important;
    }
    .panic-button>button:hover {
        background-color: #662222 !important;
        border-color: #ff4444 !important;
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.4) !important;
    }
    .status-badge {
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA BRIDGING ---
STATE_FILE = "data/live_state.json"
CMD_FILE = "data/app_commands.json"

def load_live_state():
    if not os.path.exists(STATE_FILE): return None
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except: return None

def send_command(action, params=None):
    commands = []
    if os.path.exists(CMD_FILE):
        try:
            with open(CMD_FILE, 'r') as f:
                commands = json.load(f)
        except: pass
    
    commands.append({"action": action, "params": params or {}})
    with open(CMD_FILE, 'w') as f:
        json.dump(commands, f)
    st.toast(f"🚀 Command Sent: {action}", icon="⚔️")

# --- 3. UI COMPONENTS ---

def render_sidebar(state):
    st.sidebar.title("🛡️ NEXUS C2")
    if state:
        ts = datetime.fromisoformat(state.get("timestamp", datetime.now().isoformat()))
        latency = (datetime.now() - ts).total_seconds()
        status_color = "#00ffcc" if latency < 5 else "#ffcc00" if latency < 15 else "#ff4444"
        st.sidebar.markdown(f"""
            <div style="background:{status_color}22; border:1px solid {status_color}; padding:10px; border-radius:5px;">
                <small>SYSTEM STATUS</small><br>
                <b style="color:{status_color}">{"OPERATIONAL" if latency < 10 else "DÉGRADÉ"}</b><br>
                <small>Latency: {latency:.1f}s</small>
            </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.divider()
    st.sidebar.subheader("War Room Controls")
    
    risk = st.sidebar.slider("Risk Factor (Aggression)", 0.1, 5.0, float(state.get("risk_factor", 1.0)) if state else 1.0, 0.1)
    if st.sidebar.button("Update Risk Profile"):
        send_command("SET_RISK_FACTOR", {"value": risk})
    
    st.sidebar.divider()
    if st.sidebar.button("Reload Neural Weights"):
        send_command("RELOAD_MODELS")
        
    st.sidebar.divider()
    st.sidebar.markdown('<div class="panic-button">', unsafe_allow_html=True)
    if st.sidebar.button("🚨 EMERGENCY CLOSE ALL"):
        send_command("PANIC_CLOSE_ALL")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def render_live_ops(state):
    if not state:
        st.error("📡 Waiting for Nexus Core...")
        return

    acc = state.get("account", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EQUITY", f"${acc.get('equity', 0):,.2f}")
    c2.metric("FLOATING PNL", f"${acc.get('pnl', 0):,.2f}", delta_color="normal")
    c3.metric("ACTIVE POS", state.get("positions_count", 0))
    c4.metric("MARGIN LEVEL", f"{acc.get('margin_level', 0):.0f}%")

    st.divider()
    
    # Risk Alerts
    cb = state.get("circuit_breaker", {})
    if cb.get("status") == "TRIPPED":
        st.error(f"⚠️ CIRCUIT BREAKER TRIPPED | Latency: {cb.get('latency', 0):.2f}s")
    else:
        st.success("✅ Circuit Breaker: SECURE")

def render_neural_tab(state):
    if not state: return
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Neural Confidence")
        preds = state.get("predictions", {})
        p_buy = preds.get("buy", 0.0)
        p_sell = preds.get("sell", 0.0)
        p_hold = preds.get("hold", 0.0)
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = max(p_buy, p_sell) * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "BUY" if p_buy > p_sell else "SELL"},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "#00ffcc" if p_buy > p_sell else "#ff4444"},
                'bgcolor': "#1a1f26",
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 80}
            }
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Physics Vectors
        k_ctx = state.get("kinetic_context", {})
        st.markdown(f"**Pattern:** `{k_ctx.get('label', 'UNKNOWN')}`")
        st.progress(min(k_ctx.get("coherence", 0.0), 1.0))
        st.caption(f"Kinetic Coherence: {k_ctx.get('coherence', 0.0):.2f}")

    with col_right:
        st.subheader("Neural Cortex Heatmap")
        map_data = state.get("spatial_map", [])
        if map_data:
            z_data = np.array(map_data).squeeze()
            fig_map = px.imshow(
                z_data,
                color_continuous_scale='Magma',
                origin='lower'
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=400
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Gathering spatial tensors...")

def render_war_room_tab(state):
    st.subheader("Operational Management")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### Protocol Overrides")
        if st.button("Manual Heartbeat"):
            st.toast("Heartbeat forced.")
        
        st.markdown("---")
        st.markdown("### Neural Weight Management")
        weights_dir = "app/ml/weights"
        if os.path.exists(weights_dir):
            files = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]
            selected_model = st.selectbox("Active Weight File", files)
            if st.button("Hot-Reload Weight"):
                send_command("RELOAD_MODELS", {"file": selected_model})
    
    with c2:
        st.markdown("### Risk Sentinel Adjustments")
        st.info("Current Mode: ASYMMETRIC WARFARE")
        st.metric("Risk per Trade", f"{state.get('risk_factor', 1.0)/10:.1f}%")
        st.markdown("---")
        st.markdown("### System Logs (Live)")
        log_file = "logs/mcp_server.log"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = f.readlines()[-20:]
                st.code("".join(logs), language="text")

# --- 4. MAIN APP LOOP ---

state = load_live_state()
render_sidebar(state)

st.title("FEAT NEXUS // COMMAND & CONTROL")

tab1, tab2, tab3 = st.tabs(["📡 LIVE OPERATIONS", "🧠 NEURAL CORTEX", "⚔️ WAR ROOM"])

with tab1:
    render_live_ops(state)

with tab2:
    render_neural_tab(state)

with tab3:
    render_war_room_tab(state)

# Auto-refresh logic
st.empty()
time.sleep(1)
if st.button("RECARGAR DASHBOARD"):
    st.rerun()
st.caption("Auto-refreshing in 1s...")
st.markdown("""<script>setTimeout(function(){location.reload();}, 2000);</script>""", unsafe_allow_html=True)
