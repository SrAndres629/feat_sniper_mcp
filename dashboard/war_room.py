import streamlit as st
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Add root directory to path to reach app module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.config import settings

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
STATE_FILE = settings.DASHBOARD_LIVE_STATE_PATH
API_BASE_URL = "http://localhost:8000"

def load_live_state():
    if not os.path.exists(STATE_FILE): return None
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except: return None

def api_call(method: str, endpoint: str, data: dict = None) -> dict:
    """Make API call to the Mission Control backend."""
    import requests
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=data or {}, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API Server not connected. Using fallback mode.")
        return {}
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}

def send_command(action, params=None):
    """Send command via API (with file-based fallback)."""
    endpoint_map = {
        "START_SIMULATION": ("/api/simulation/start", "POST"),
        "STOP_SIMULATION": ("/api/simulation/stop", "POST"),
        "SET_RISK_FACTOR": ("/api/risk/update", "POST"),
        "PANIC_CLOSE_ALL": ("/api/emergency/close-all", "POST"),
        "RELOAD_MODELS": ("/api/models/reload", "POST"),
    }
    
    if action in endpoint_map:
        endpoint, method = endpoint_map[action]
        result = api_call(method, endpoint, params)
        if result.get("success"):
            st.toast(f"✅ {result.get('message', action)}", icon="⚔️")
        return result
    
    # Fallback to file-based for unknown commands
    CMD_FILE = settings.DASHBOARD_COMMAND_PATH
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
    
    # MACRO SENTINEL: News/DEFCON Status
    macro = state.get("macro_status", {})
    if macro:
        defcon = macro.get("defcon_name", "DEFCON_5")
        minutes_until = macro.get("minutes_until", float("inf"))
        next_event = macro.get("next_event_name", "None")
        position_mult = macro.get("position_multiplier", 1.0)
        kill_switch = macro.get("kill_switch", False)
        
        st.divider()
        st.subheader("📡 MACRO SENTINEL")
        
        mc1, mc2, mc3 = st.columns(3)
        
        # DEFCON Color Coding
        if "DEFCON_1" in defcon:
            mc1.error(f"🚨 {defcon}")
        elif "DEFCON_2" in defcon:
            mc1.warning(f"⚠️ {defcon}")
        elif "DEFCON_3" in defcon:
            mc1.info(f"📢 {defcon}")
        else:
            mc1.success(f"✅ {defcon}")
        
        # Next Event
        if minutes_until != float("inf"):
            mc2.metric("Next Event", next_event, f"in {minutes_until:.0f}m")
        else:
            mc2.metric("Next Event", "None", "Clear horizon")
        
        # Position Multiplier
        mc3.metric("Position Mult", f"{position_mult:.0%}")
        
        # Kill Switch Warning
        if kill_switch:
            st.error("⛔ KILL SWITCH ACTIVE: NO NEW TRADES ALLOWED")


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
        weights_dir = settings.MODELS_DIR
        if os.path.exists(weights_dir):
            files = [f for f in os.listdir(weights_dir) if f.endswith(".pt")]
            selected_model = st.selectbox("Active Weight File", files)
            if st.button("Hot-Reload Weight"):
                send_command("RELOAD_MODELS", {"file": selected_model})
    
    with c2:
        st.markdown("### Risk Sentinel Adjustments")
        st.info("Current Mode: ASYMMETRIC WARFARE")
        st.metric("Risk per Trade", f"{state.get('risk_factor', 1.0)/10:.1f}%")
        st.markdown("---")
        st.markdown("### System Logs (Live)")
        log_file = "logs/nexus_daemon.log"
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = f.readlines()[-20:]
                    st.code("".join(logs), language="text")
            except: 
                st.code("Logs currently unavailable.", language="text")

def render_training_arena():
    """Training Arena Tab - Control simulation training from UI."""
    st.subheader("🎓 Training Arena")
    st.markdown("Control neural network training simulations directly from the dashboard.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Simulation Configuration")
        episodes = st.number_input("Episodes", min_value=5, max_value=1000, value=50, step=5)
        st.caption("More episodes = Better training, but takes longer.")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Start Simulation", use_container_width=True):
                send_command("START_SIMULATION", {"episodes": episodes})
                st.success(f"Simulation started with {episodes} episodes!")
        with c2:
            if st.button("🛑 Stop Simulation", use_container_width=True):
                send_command("STOP_SIMULATION")
                st.warning("Stop command sent.")
    
    with col2:
        st.markdown("### Simulation Status")
        
        # Fetch status from API
        sim_status = api_call("GET", "/api/simulation/status")
        
        if sim_status:
            current_ep = sim_status.get("current_episode", 0)
            total_ep = sim_status.get("total_episodes", 1) or 1
            balance = sim_status.get("current_balance", 20.0)
            running = sim_status.get("running", False)
            elapsed = sim_status.get("elapsed_seconds", 0)
            remaining = sim_status.get("estimated_remaining_seconds", 0)
            
            if running:
                st.info(f"🏃 Running: Episode {current_ep}/{total_ep}")
                st.progress(current_ep / total_ep if total_ep > 0 else 0)
                st.caption(f"⏱️ Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
            else:
                st.success("✅ Simulation Complete" if current_ep > 0 else "⏸️ Idle")
            
            st.metric("Last Balance", f"${balance:.2f}")
        else:
            st.info("Connecting to API...")

def render_analytics_tab():
    """Analytics Tab - Display historical performance metrics."""
    st.subheader("📊 Performance Analytics")
    
    journal_path = "data/trade_journal.json"
    
    if not os.path.exists(journal_path):
        st.warning("No trade journal found. Run simulations or live trades to generate data.")
        return
    
    try:
        with open(journal_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        closed = [e for e in entries if e.get("status") == "CLOSED"]
        
        if not closed:
            st.info("No closed trades yet. Data will appear after trades close.")
            return
        
        # Calculate Metrics
        wins = [e for e in closed if e.get("result") == "WIN"]
        losses = [e for e in closed if e.get("result") == "LOSS"]
        
        win_rate = (len(wins) / len(closed)) * 100 if closed else 0
        total_pnl = sum(e.get("pnl_pips", 0) for e in closed)
        avg_win = sum(e.get("pnl_pips", 0) for e in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(e.get("pnl_pips", 0) for e in losses) / len(losses)) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Display Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", f"{win_rate:.1f}%")
        c2.metric("Total PnL (pips)", f"{total_pnl:.1f}")
        c3.metric("Profit Factor", f"{profit_factor:.2f}")
        c4.metric("Total Trades", len(closed))
        
        st.divider()
        
        # Equity Curve
        st.markdown("### Equity Curve")
        if closed:
            equity = [20.0]  # Starting balance
            for e in closed:
                equity.append(equity[-1] + e.get("pnl_pips", 0) * 0.1)  # Simplified conversion
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=equity, mode='lines', name='Equity', line=dict(color='#00ffcc')))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Exit Reason Breakdown
        st.markdown("### Exit Reasons")
        exit_stats = {}
        for e in closed:
            reason = e.get("exit_reason", "UNKNOWN")
            exit_stats[reason] = exit_stats.get(reason, 0) + 1
        
        if exit_stats:
            fig_pie = px.pie(names=list(exit_stats.keys()), values=list(exit_stats.values()))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_pie, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# --- 4. MAIN APP LOOP ---

state = load_live_state()
render_sidebar(state)

st.title("FEAT NEXUS // COMMAND & CONTROL")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📡 LIVE OPS", "🧠 NEURAL", "⚔️ WAR ROOM", "🎓 TRAINING", "📊 ANALYTICS"])

with tab1:
    render_live_ops(state)

with tab2:
    render_neural_tab(state)

with tab3:
    render_war_room_tab(state)

with tab4:
    render_training_arena()

with tab5:
    render_analytics_tab()

# Auto-refresh logic
st.empty()
time.sleep(settings.DASHBOARD_REFRESH_SLEEP_SEC)
if st.button("RECARGAR DASHBOARD"):
    st.rerun()
st.caption(f"Auto-refreshing in {settings.DASHBOARD_REFRESH_SLEEP_SEC}s...")
st.markdown(f"""<script>setTimeout(function(){{location.reload();}}, {settings.DASHBOARD_REFRESH_JS_MS});</script>""", unsafe_allow_html=True)
