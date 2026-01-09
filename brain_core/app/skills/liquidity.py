import logging
import MetaTrader5 as mt5
from typing import Dict, Any, List
from app.core.mt5_conn import mt5_conn
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Skills.Liquidity")

class LiquidityEngine:
    """
    Sniper 2.0 Liquidity Analysis.
    Analyzes Depth of Market (DoM) to ensure institutional fill quality.
    """

    @staticmethod
    async def get_market_depth(symbol: str) -> Dict[str, Any]:
        """
        Retrieves and analyzes the Depth of Market (if supported by broker).
        Instrumentado con OTel.
        """
        from app.core.observability import tracer
        
        with tracer.start_as_current_span("mt5_liquidity_depth") as span:
            span.set_attribute("symbol", symbol)
            
            # Attempt to subscribe to DoM
            await mt5_conn.execute(mt5.market_book_add, symbol)
            
            # Get current book
            book = await mt5_conn.execute(mt5.market_book_get, symbol)
            
            if not book:
                return {
                    "status": "warning",
                    "message": "DoM not supported or empty for this symbol.",
                    "liquidity_score": 0.5 
                }

            # Analyze liquidity depth
            total_bid_vol = sum([level.volume for level in book if level.type == mt5.BOOK_TYPE_SELL])
            total_ask_vol = sum([level.volume for level in book if level.type == mt5.BOOK_TYPE_BUY])
            
            liquidity_usd = (total_bid_vol + total_ask_vol) * 100000 
            score = min(1.0, liquidity_usd / settings.MIN_LIQUIDITY_DEPTH)

            span.set_attribute("liquidity_score", score)

            return {
                "status": "success",
                "symbol": symbol,
                "bid_volume": int(total_bid_vol),
                "ask_volume": int(total_ask_vol),
                "total_liquidity_est": round(liquidity_usd, 2),
                "institutional_grade": liquidity_usd >= settings.MIN_LIQUIDITY_DEPTH,
                "liquidity_score": round(score, 2)
            }

async def check_liquidity_preflight(symbol: str) -> bool:
    """Institutional Guard: Pre-flight check for liquidity."""
    analysis = await LiquidityEngine.get_market_depth(symbol)
    if analysis["liquidity_score"] < 0.3:
        logger.warning(f"Sniper 2.0: Liquidity too low for {symbol} ({analysis['liquidity_score']}). Vetoing entry.")
        return False
    return True

liquidity_engine = LiquidityEngine()
