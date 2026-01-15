import numpy as np
import pandas as pd
import logging
import MetaTrader5 as mt5
from typing import Dict, Any, List, Optional
from app.core.mt5_conn import mt5_conn
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Skills.Liquidity")

class LiquidityGrid:
    """
    Gate E: Liquidity Grid mapping.
    Identifies institutional levels (PDH/PDL) and structural imbalances (FVG).
    """
    def __init__(self):
        print("[Liquidity] Grid Built (PDH/PDL calculation)")
        self.levels = {}

    def calculate_pdh_pdl(self, daily_candles: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates Previous Day High (PDH) and Previous Day Low (PDL).
        Input: daily_candles DataFrame with 'high', 'low' columns.
        """
        if daily_candles.empty or len(daily_candles) < 1:
            return {"pdh": 0.0, "pdl": 0.0}
        
        # Last confirmed daily candle (i-1)
        last_day = daily_candles.iloc[-1]
        self.levels["pdh"] = float(last_day['high'])
        self.levels["pdl"] = float(last_day['low'])
        
        logger.info(f"📊 PDH/PDL Mapped: {self.levels['pdh']} / {self.levels['pdl']}")
        return self.levels

    def detect_fvg(self, candle_series: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detects Fair Value Gaps (FVG) in the provided series.
        Candle(i) is FVG if:
        Bullish: Low(i) > High(i-2)
        Bearish: High(i) < Low(i-2)
        """
        fvgs = []
        if len(candle_series) < 3:
            return fvgs

        for i in range(2, len(candle_series)):
            # Bullish FVG
            if candle_series.iloc[i]['low'] > candle_series.iloc[i-2]['high']:
                fvgs.append({
                    "type": "BULLISH_FVG",
                    "top": candle_series.iloc[i]['low'],
                    "bottom": candle_series.iloc[i-2]['high'],
                    "index": i
                })
            # Bearish FVG
            elif candle_series.iloc[i]['high'] < candle_series.iloc[i-2]['low']:
                fvgs.append({
                    "type": "BEARISH_FVG",
                    "top": candle_series.iloc[i-2]['low'],
                    "bottom": candle_series.iloc[i]['high'],
                    "index": i
                })
        
        self.levels["fvgs"] = fvgs
        return fvgs

liquidity_grid = LiquidityGrid()

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
