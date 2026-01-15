import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("feat.trade_sentinel")

class TradeSentinel:
    """
    NEURAL SENTINEL (The Guardian)
    Monitors active trades for 'Vital Signs' decay.
    Authority: CAN CLOSE TRADES IMMEDIATELY.
    """
    def __init__(self, zmq_bridge):
        self.zmq_bridge = zmq_bridge
        self.monitored_trades = {} # {ticket: {entry_p_win, entry_time, last_physics}}
        logger.info("[SENTINEL] Neural Guardian Online - Watching for Confidence Crashes")

    async def register_trade(self, ticket: int, p_win: float, physics_state: Any):
        """Start monitoring a new trade."""
        self.monitored_trades[ticket] = {
            "entry_p_win": p_win,
            "entry_time": time.time(),
            "last_physics": physics_state,
            "max_p_win": p_win
        }
        logger.info(f"👁️ SENTINEL: Monitoring Ticket {ticket} (Initial p_win: {p_win:.2f})")

    async def check_vital_signs(self, ticket: int, current_p_win: float, current_physics: Any) -> Dict[str, Any]:
        """
        Evaluates trade health. Returns action: 'HOLD', 'CLOSE_NOW', 'TIGHTEN_SL'.
        """
        if ticket not in self.monitored_trades:
            return {"action": "HOLD", "reason": "Not monitored"}

        trade_data = self.monitored_trades[ticket]
        entry_p_win = trade_data["entry_p_win"]
        
        # Track max confidence reached
        if current_p_win > trade_data["max_p_win"]:
            trade_data["max_p_win"] = current_p_win

        # RULE 1: CONFIDENCE CRASH (Neural Early Close)
        # If confidence drops significantly below entry or absolute threshold of 0.45
        if current_p_win < 0.45:
             logger.warning(f"🚨 SENTINEL: Ticket {ticket} CONFIDENCE CRASH ({current_p_win:.2f} < 0.45). Signal CLOSE.")
             return {"action": "CLOSE_NOW", "reason": "Confidence Crash"}
        
        if (trade_data["max_p_win"] - current_p_win) > 0.20:
             logger.warning(f"📉 SENTINEL: Ticket {ticket} Momentum Lost (Max {trade_data['max_p_win']:.2f} -> Curr {current_p_win:.2f}). Signal CLOSE.")
             return {"action": "CLOSE_NOW", "reason": "Momentum Decay > 20%"}

        # RULE 2: PHYSICS REVERSAL (if physics object has directional cues)
        # Placeholder for complex physics check
        
        return {"action": "HOLD", "reason": "Vital signs stable"}

    def unregister_trade(self, ticket: int):
        if ticket in self.monitored_trades:
            del self.monitored_trades[ticket]
