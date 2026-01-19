import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from app.core.config import settings

logger = logging.getLogger("feat.pos_sentinel")

class PositionSentinel:
    """
    [PHASE 13 - DOCTORAL PROTECTION]
    Monitors active positions for Exhaustion (Physics) and Time-Limits.
    """
    def __init__(self, trade_manager):
        self.tm = trade_manager
        self.exhaustion_atr_threshold = settings.EXHAUSTION_ATR_THRESHOLD
        self.scalp_time_limit_seconds = settings.SCALP_TIME_LIMIT_SECONDS

    async def check_exhaustion_exit(self, ticket: int, current_price: float, 
                                     current_l4_slope: float, current_time: datetime) -> Dict:
        """
        Exhaustion Exit Logic: Triggers early exit or breakeven based on physics.
        """
        if ticket not in self.tm.active_positions:
            return {"status": "NO_POSITION", "ticket": ticket}
        
        pos = self.tm.active_positions[ticket]
        entry_price = pos.get("entry_price", current_price)
        atr = pos.get("atr", 0.0)
        open_time = pos.get("open_time", current_time)
        is_buy = pos.get("is_buy", True)
        
        # Calculate current profit in ATR units
        price_diff = (current_price - entry_price) if is_buy else (entry_price - current_price)
        atr_profit = price_diff / (atr + 1e-9)
        
        # Time elapsed
        time_elapsed = (current_time - open_time).total_seconds()
        result = {"status": "HOLDING", "ticket": ticket, "atr_profit": round(atr_profit, 2)}
        
        # Rule 1: Exhaustion Exit (0.5 ATR profit + Slope dying)
        if atr_profit >= self.exhaustion_atr_threshold:
            if current_l4_slope < 0:
                # Physics exhaustion detected - Move to Breakeven
                pip_val = 0.01 if "XAU" in pos.get("symbol", "") or "JPY" in pos.get("symbol", "") else 0.0001
                new_sl = entry_price + (pip_val if is_buy else -pip_val)
                logger.info(f"🛡️ EXHAUSTION EXIT: Moving SL to Breakeven for Ticket {ticket}")
                await self.tm.modify_position(ticket, sl=new_sl, tp=pos.get("tp", 0))
                result["status"] = "BREAKEVEN_SET"
        
        # Rule 2: Time-Limit Exit
        if time_elapsed > self.scalp_time_limit_seconds:
            if abs(current_l4_slope) < 0.01:
                logger.warning(f"⏰ TIME-LIMIT EXIT: Ticket {ticket} closed (Elapsed: {time_elapsed:.0f}s)")
                await self.tm.close_position(ticket)
                result["status"] = "TIME_EXIT"
        
        return result

    async def update_positions_logic(self, probability: float):
        """Checks probability decay to close losing trades early."""
        if not self.tm.active_positions: return

        to_close = []
        for ticket, pos in self.tm.active_positions.items():
            is_buy = pos.get("is_buy", True)
            if (is_buy and probability < 0.40) or (not is_buy and probability > 0.60):
                 to_close.append(ticket)
                 
        for ticket in to_close:
             logger.info(f"⚡ ACTIVE MGMT: Probability Decay Veto on Ticket {ticket}")
             await self.tm.close_position(ticket)
