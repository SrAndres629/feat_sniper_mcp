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

interface = MCPInterface()

@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    logger.info("📡 MCP Interface: Online (Diplomatic Layer Active)")
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

# ... Additional tools would be refactored here to follow this pattern ...

if __name__ == "__main__":
    mcp.run()
