import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("feat.neuro.liquidity")

class LiquidityChannel:
    """
    🧬 Neuro-Channel: Liquidity & Order Flow
    Identifies 'Where the money is' using OB, FVG, and PVP density.
    """
    def __init__(self):
        from nexus_core.structure_engine import structure_engine
        self.structure_engine = structure_engine

    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        # Logic to extract institutional zones from structure_engine
        zones = self.structure_engine.detect_zones(df)
        last_row = zones.iloc[-1]
        
        blocks = []
        if last_row['zone_type'] != 'NONE':
            blocks.append({
                "type": last_row['zone_type'],
                "price": last_row['zone_low'] if last_row['zone_type'] == 'DEMAND' else last_row['zone_high'],
                "strength": last_row['zone_strength']
            })
        return blocks

    def get_liquidity_bias(self, df: pd.DataFrame) -> float:
        """Returns 1.0 (Bullish Liquidity) to -1.0 (Bearish Liquidity)."""
        health = self.structure_engine.get_structural_health(df)
        # Map status to bias
        status_map = {"HEALTHY_BULL": 1.0, "WEAK_BULL": 0.5, "NEUTRAL": 0.0, "WEAK_BEAR": -0.5, "HEALTHY_BEAR": -1.0}
        return status_map.get(health["status"], 0.0)

    def compute_fvg(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identifies Fair Value Gaps for neural ingestion."""
        if len(df) < 3: return []
        gaps = []
        # Vectorized detection of FVG
        highs = df['high'].values
        lows = df['low'].values
        
        # Bullish FVG: Low of candle 3 > High of candle 1
        bull_fvg = (lows[2:] > highs[:-2])
        # Bearish FVG: High of candle 3 < Low of candle 1
        bear_fvg = (highs[2:] < lows[:-2])
        
        # We only care about the most recent ones for scalping
        if bull_fvg[-1]: gaps.append({"type": "BULL_FVG", "top": lows[-1], "bottom": highs[-3]})
        if bear_fvg[-1]: gaps.append({"type": "BEAR_FVG", "top": lows[-3], "bottom": highs[-1]})
        
        return gaps
