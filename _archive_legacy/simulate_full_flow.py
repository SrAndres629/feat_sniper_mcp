import asyncio
import logging
import sys
import os
import json
from unittest.mock import MagicMock, AsyncMock

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("SIMULATION")

# Mock MT5
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5

# Patch ZMQ Bridge BEFORE importing modules that use it
from app.core import zmq_bridge
zmq_bridge.zmq_bridge.sub_socket = MagicMock()
zmq_bridge.zmq_bridge.pub_socket = MagicMock()
# Mock send_string as async
zmq_bridge.zmq_bridge.pub_socket.send_string = AsyncMock()

# Patch ML Engine
from app.ml import ml_engine
ml_engine.ml_engine.predict_next_tick = AsyncMock(return_value={
    "p_win": 0.92,
    "confidence": 0.95, 
    "direction": 1, # BUY
    "volatility": 0.005,
    "urgency": 0.85 # High Urgency
})

# Patch N8N Bridge
from app.services import n8n_bridge
n8n_bridge.n8n_bridge.request_audit = AsyncMock(return_value=MagicMock(
    decision="APPROVE",
    feedback="Simulation Approved",
    suggested_sl=None,
    suggested_tp=None
))

# Import Commander
from nexus_control import NexusControl, StrategyCommander, TradingState

async def run_simulation():
    print("\n🕵️ MODEL 5: END-TO-END DRY RUN SIMULATION")
    print("===========================================")
    
    # Instantiate Commander
    commander = StrategyCommander()
    
    # Inject Synthetic Data via ZMQ Mock
    synthetic_tick = {
        "symbol": "XAUUSD",
        "bid": 2030.0,
        "ask": 2030.5,
        "time": 1700000000,
        "flags": 6
    }
    
    # We will invoke execute_logic directly rather than run the full loop to avoid infinite loop
    # But we want to simulate the "Flow".
    
    print("[1] INJECTING MARKET DATA (TICK)...")
    logger.info("ZMQ: Tick Received: XAUUSD @ 2030.5")
    
    # Execute Logic Step
    await commander.execute_logic(synthetic_tick)
    
    print("\n✅ SIMULATION COMPLETE. CHECK LOGS ABOVE FOR SEQUENCE.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
