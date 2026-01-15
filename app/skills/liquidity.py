import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger("feat.liquidity")

class LiquidityMap:
    """
    [E] COMPONENT - SPACE (The Liquidity Grid)
    Maps institutional liquidity pools (PDH, PDL) and Imbalances (FVG).
    """
    def __init__(self):
        logger.info("[Liquidity] Map Engine Online (Mapping Pools & Imbalances)")
        self.pools = {"pdh": 0.0, "pdl": 0.0, "pwh": 0.0, "pwl": 0.0}
        self.imbalances: List[Dict[str, Any]] = []

    def refresh_grid(self, h1_df: pd.DataFrame, d1_df: Optional[pd.DataFrame] = None):
        """
        Refreshes levels based on high-timeframe data.
        """
        if d1_df is not None and len(d1_df) >= 2:
            last_day = d1_df.iloc[-2] # Previous day completed
            self.pools["pdh"] = float(last_day['high'])
            self.pools["pdl"] = float(last_day['low'])
        elif len(h1_df) >= 24:
            # Proxy PDH/PDL from last 24 H1 candles
            self.pools["pdh"] = float(h1_df['high'].iloc[-24:].max())
            self.pools["pdl"] = float(h1_df['low'].iloc[-24:].min())

        self.imbalances = self._detect_unmitigated_fvgs(h1_df)
        
        logger.debug(f"[SPACE] PDH: {self.pools['pdh']} | PDL: {self.pools['pdl']} | FVGs: {len(self.imbalances)}")

    def _detect_unmitigated_fvgs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detects Fair Value Gaps that have not been filled/mitigated.
        """
        fvgs = []
        if len(df) < 5: return fvgs

        for i in range(2, len(df)):
            # Bullish FVG
            is_bull_fvg = df['low'].iloc[i] > df['high'].iloc[i-2]
            if is_bull_fvg:
                top = df['low'].iloc[i]
                bottom = df['high'].iloc[i-2]
                # Mitigation check: have any subsequent candles touched this zone?
                # For simplicity, we check candles from i+1 to end
                mitigated = (df['low'].iloc[i+1:] < top).any() if i+1 < len(df) else False
                if not mitigated:
                    fvgs.append({"type": "BULL_FVG", "top": top, "bottom": bottom, "price": (top+bottom)/2})
            
            # Bearish FVG
            is_bear_fvg = df['high'].iloc[i] < df['low'].iloc[i-2]
            if is_bear_fvg:
                top = df['low'].iloc[i-2]
                bottom = df['high'].iloc[i]
                mitigated = (df['high'].iloc[i+1:] > bottom).any() if i+1 < len(df) else False
                if not mitigated:
                    fvgs.append({"type": "BEAR_FVG", "top": top, "bottom": bottom, "price": (top+bottom)/2})
        
        return fvgs

    def get_nearest_poi(self, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Returns the closest institutional level / imbalance.
        """
        all_levels = []
        for name, val in self.pools.items():
            if val > 0: all_levels.append({"name": name, "price": val})
        for fvg in self.imbalances:
            all_levels.append({"name": fvg["type"], "price": fvg["price"]})
            
        if not all_levels: return None
        
        # Sort by distance to current price
        all_levels.sort(key=lambda x: abs(x["price"] - current_price))
        return all_levels[0]

# Global singleton
liquidity_map = LiquidityMap()

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
