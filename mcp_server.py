"""
FEAT NEXUS: DIPLOMATIC INTERFACE (MCP)
======================================
This node handles communication between Claude/Agent and the Immortal Core.
It provides tools for observation and manual overrides, but the Core 
operates independently in nexus_daemon.py.
"""

import asyncio
import logging
import os
import json
import pandas as pd
from contextlib import asynccontextmanager
from fastmcp import FastMCP
from app.core.config import settings

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [MCP] | %(message)s'
)
logger = logging.getLogger("nexus.mcp")

# --- INTERFACE STATE ---
mcp = FastMCP("FEAT_Sniper_C2")

class MCPInterface:
    def __init__(self):
        self.state_file = "data/live_state.json"
        self.cmd_file = "data/app_commands.json"

    def get_core_state(self):
        if not os.path.exists(self.state_file):
            return {"status": "OFFLINE", "reason": "Live state file not found."}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}

    def send_command(self, action: str, params: dict = None):
        commands = []
        if os.path.exists(self.cmd_file):
            try:
                with open(self.cmd_file, 'r') as f:
                    commands = json.load(f)
            except: pass
        commands.append({"action": action, "params": params or {}})
        with open(self.cmd_file, 'w') as f:
            json.dump(commands, f)
        return True

from app.core.mt5_conn.manager import MT5Connection

interface = MCPInterface()

async def passive_mt5_monitor():
    """Background task for Standalone Mode."""
    logger.info("🛡️ PASSIVE MONITOR: Initializing Read-Only MT5 Link...")
    conn = MT5Connection()
    if await conn.startup():
        logger.info("✅ PASSIVE MONITOR: Connected. Polling market data...")
        while True:
            try:
                # Update State for Tools
                info = await conn.get_account_info()
                state = {
                    "status": "PASSIVE_MODE",
                    "symbol": settings.SYMBOL if hasattr(settings, 'SYMBOL') else "XAUUSD",
                    "account": {
                        "balance": info.get("balance", 0.0),
                        "equity": info.get("equity", 0.0),
                        "pnl": round(info.get("equity", 0.0) - info.get("balance", 0.0), 2)
                    },
                    "note": "Trading Engine OFFLINE. Tools are in Observation Mode."
                }
                # Write to disk so tools can read it
                with open(interface.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
            except Exception as e:
                logger.error(f"Monitor Loop Error: {e}")
            
            await asyncio.sleep(5)
    else:
        logger.error("❌ PASSIVE MONITOR: MT5 Connection Failed.")

@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    logger.info("📡 MCP Interface: Online (Diplomatic Layer Active)")
    
    # [STANDALONE MODE CHECK]
    if os.environ.get("FEAT_MCP_STANDALONE") == "1":
        logger.warning("⚠️ RUNNING IN STANDALONE MODE - NO TRADING ENGINE")
        task = asyncio.create_task(passive_mt5_monitor())
        yield
        task.cancel()
    else:
        yield
        
    logger.info("👋 MCP Interface: Shutdown.")

# --- TOOLS (REFACTORED FOR DECOUPLED OPS) ---

@mcp.tool()
async def get_system_status() -> str:
    """Returns the full status of the Immortal Core and account metrics."""
    state = interface.get_core_state()
    return json.dumps(state, indent=2)

@mcp.tool()
async def set_risk_profile(risk_factor: float) -> str:
    """Adjusts the bot's risk aggression (0.1 to 5.0)."""
    if not (0.1 <= risk_factor <= 5.0):
        return "❌ Error: Risk factor must be between 0.1 and 5.0"
    interface.send_command("SET_RISK_FACTOR", {"value": risk_factor})
    return f"✅ Command Sent: Set Risk Factor to {risk_factor}. Pending Core acknowledgement."

@mcp.tool()
async def trigger_panic_close() -> str:
    """EMERGENCY: Closes all active positions immediately."""
    interface.send_command("PANIC_CLOSE_ALL")
    return "🚨 EMERGENCY COMMAND SENT: Closing all positions. Check status in 5s."

@mcp.tool()
async def get_market_analysis() -> str:
    """Summarizes current market regime and neural confidence from the last state."""
    state = interface.get_core_state()
    if state.get("status") == "OFFLINE":
        return "❌ Core is offline. No live analysis available."
    
    analysis = {
        "symbol": state.get("symbol"),
        "neural_confidence": state.get("win_confidence"),
        "regime": state.get("kinetic_context", {}).get("label"),
        "coherence": state.get("kinetic_context", {}).get("coherence"),
        "pnl_floating": state.get("account", {}).get("pnl")
    }
    return json.dumps(analysis, indent=2)

@mcp.tool()
async def reload_brain() -> str:
    """Forces the Immortal Core to reload neural weights from disk (hot-swap)."""
    interface.send_command("RELOAD_MODELS")
    return "🧠 RELOAD COMMAND SENT: Synchronizing AI weights. Check logs for confirmation."

@mcp.tool()
async def get_performance_report() -> str:
    """Institutional Analytics: Generates Sharpe Ratio, Win Rate, and P&L metrics."""
    journal_path = "data/trade_journal.json"
    if not os.path.exists(journal_path):
        return "❌ Error: Trade journal not found. No performance data available."
    
    try:
        with open(journal_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        closed = [e for e in entries if e.get("status") == "CLOSED"]
        if not closed:
            return "📈 Status: No closed trades recorded yet."
        
        wins = [e for e in closed if e.get("result") == "WIN"]
        wr = (len(wins) / len(closed)) * 100
        pnl = sum(e.get("pnl_pips", 0) for e in closed)
        
        report = {
            "total_trades": len(closed),
            "win_rate": f"{wr:.2f}%",
            "net_pnl_pips": round(pnl, 2),
            "avg_duration_min": round(sum(e.get("duration_minutes", 0) for e in closed) / len(closed), 1),
            "exit_stats": {}
        }
        
        for e in closed:
            reason = e.get("exit_reason", "UNKNOWN")
            report["exit_stats"][reason] = report["exit_stats"].get(reason, 0) + 1
            
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"❌ Error analyzing journal: {str(e)}"

if __name__ == "__main__":
    mcp.run()
