"""
ML Normalization - Cross-Asset Intelligence Layer
==================================================
Converts raw market data to ATR-normalized features.
This enables transfer learning across different assets.

Key concept: A "1 ATR" move means the same thing whether
it's XAUUSD (Gold) or EURUSD (Euro).
"""

import numpy as np
from typing import Dict, Optional


class ATRNormalizer:
    """
    Normalizes market data using ATR (Average True Range).
    
    Benefits:
    - Asset-agnostic features (model trained on Gold works on Euro)
    - Scale-invariant signals (pip differences become irrelevant)
    - Better generalization across volatility regimes
    """
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self._atr_cache: Dict[str, float] = {}
        self._volume_ma_cache: Dict[str, float] = {}
        
    def set_current_atr(self, symbol: str, atr: float):
        """Set the current ATR for normalization."""
        self._atr_cache[symbol] = atr
        
    def set_volume_ma(self, symbol: str, volume_ma: float):
        """Set the 20-period volume moving average."""
        self._volume_ma_cache[symbol] = volume_ma
        
    def normalize_features(self, 
                           symbol: str,
                           features: Dict,
                           current_atr: Optional[float] = None,
                           volume_ma: Optional[float] = None) -> Dict:
        """
        Convert raw features to ATR-normalized features.
        
        Input (raw):
            close, open, high, low, volume, price_change, etc.
            
        Output (normalized):
            close_atr_ratio, candle_range_atr, volume_relative, etc.
        """
        atr = current_atr or self._atr_cache.get(symbol, 1.0)
        vol_ma = volume_ma or self._volume_ma_cache.get(symbol, 1.0)
        
        # Prevent division by zero
        if atr <= 0:
            atr = 1.0
        if vol_ma <= 0:
            vol_ma = 1.0
            
        close = features.get("close", 0)
        open_ = features.get("open", 0)
        high = features.get("high", 0)
        low = features.get("low", 0)
        volume = features.get("volume", 0)
        
        # Calculate ATR-normalized metrics
        candle_range = high - low
        candle_body = abs(close - open_)
        price_change = close - open_
        
        normalized = {
            #  Position relative to candle 
            "candle_range_atr": candle_range / atr,
            "candle_body_atr": candle_body / atr,
            "price_change_atr": price_change / atr,
            
            #  Volume analysis 
            "volume_relative": volume / vol_ma if vol_ma > 0 else 1.0,
            
            #  Volatility normalized indicators 
            "rsi": features.get("rsi", 50.0),  # RSI is already 0-100
            "ema_spread_atr": features.get("ema_spread", 0) / atr,
            
            #  FEAT metrics (normalized) 
            "feat_score": features.get("feat_score", 0) / 100.0,  # 0-1 scale
            "compression_ratio": features.get("compression", 0.5),  # Already ratio
            
            #  Zone distance 
            "liquidity_above_atr": features.get("liquidity_above", 0) / atr,
            "liquidity_below_atr": features.get("liquidity_below", 0) / atr,
            
            #  State indicators (categorical) 
            "fsm_state": features.get("fsm_state", 0),
            
            #  Original ATR for reference 
            "atr": atr
        }
        
        return normalized
        
    def normalize_stop_loss(self, desired_atr_distance: float, atr: float) -> float:
        """
        Convert ATR-based stop loss to price points.
        
        Example:
            desired_atr_distance = 1.5  # 1.5 ATR
            atr = 2.5  # Current ATR for XAUUSD
            Returns: 3.75 price points
        """
        return desired_atr_distance * atr
        
    def normalize_take_profit(self, desired_atr_distance: float, atr: float) -> float:
        """
        Convert ATR-based take profit to price points.
        
        Example:
            desired_atr_distance = 3.0  # 3 ATR (1:2 RR with 1.5 ATR stop)
            atr = 2.5  # Current ATR for XAUUSD
            Returns: 7.5 price points
        """
        return desired_atr_distance * atr


# 
# ASSET PROFILE - Metadata for Cross-Asset Intelligence
# 

ASSET_PROFILES = {
    "XAUUSD": {
        "class": "Metal",
        "pip_value": 0.01,
        "typical_atr_m5": 2.5,
        "spread_tolerance_atr": 0.1,
        "killzones": ["LONDON", "NY"],
        "correlations": ["XAGUSD", "DXY_inverse"]
    },
    "EURUSD": {
        "class": "Forex",
        "pip_value": 0.0001,
        "typical_atr_m5": 0.0015,
        "spread_tolerance_atr": 0.05,
        "killzones": ["LONDON", "NY"],
        "correlations": ["GBPUSD", "DXY_inverse"]
    },
    "BTCUSD": {
        "class": "Crypto",
        "pip_value": 1.0,
        "typical_atr_m5": 150.0,
        "spread_tolerance_atr": 0.02,
        "killzones": ["ASIA", "NY"],  # 24/7 but peaks
        "correlations": ["ETHUSD", "NASDAQ_weak"]
    },
    "US30": {
        "class": "Index",
        "pip_value": 1.0,
        "typical_atr_m5": 50.0,
        "spread_tolerance_atr": 0.05,
        "killzones": ["NY"],
        "correlations": ["SPX", "NASDAQ"]
    }
}


def get_asset_profile(symbol: str) -> Dict:
    """
    Get asset profile for normalization and context.
    Returns default profile if symbol not found.
    """
    # Try exact match
    if symbol in ASSET_PROFILES:
        return {"symbol": symbol, **ASSET_PROFILES[symbol]}
        
    # Try to detect asset class from symbol
    symbol_upper = symbol.upper()
    
    if "XAU" in symbol_upper or "GOLD" in symbol_upper:
        return {"symbol": symbol, "class": "Metal", **ASSET_PROFILES.get("XAUUSD", {})}
    elif "BTC" in symbol_upper or "ETH" in symbol_upper:
        return {"symbol": symbol, "class": "Crypto", "pip_value": 1.0, "typical_atr_m5": 100.0}
    elif any(x in symbol_upper for x in ["US30", "SPX", "NAS", "DAX"]):
        return {"symbol": symbol, "class": "Index", "pip_value": 1.0, "typical_atr_m5": 30.0}
    else:
        # Default to Forex
        return {"symbol": symbol, "class": "Forex", "pip_value": 0.0001, "typical_atr_m5": 0.001}


# Singleton instance
normalizer = ATRNormalizer()


# 
# MCP-COMPATIBLE ASYNC WRAPPERS
# 

async def get_asset_profile_tool(symbol: str) -> Dict:
    """MCP Tool: Get asset profile and metadata."""
    return get_asset_profile(symbol)


async def normalize_features_tool(symbol: str, features: Dict, atr: float) -> Dict:
    """MCP Tool: Normalize raw features to ATR-relative values."""
    return normalizer.normalize_features(symbol, features, current_atr=atr)


async def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_atr: float,
    current_atr: float,
    pip_value: float
) -> Dict:
    """
    MCP Tool: Calculate position size based on ATR risk.
    
    Args:
        account_balance: Total account balance
        risk_percent: Risk per trade (e.g., 1.0 = 1%)
        stop_loss_atr: How many ATRs for stop loss (e.g., 1.5)
        current_atr: Current ATR value
        pip_value: Value per pip for this asset
    """
    risk_amount = account_balance * (risk_percent / 100)
    stop_loss_pips = stop_loss_atr * current_atr / pip_value
    
    if stop_loss_pips <= 0:
        return {"error": "Invalid stop loss calculation"}
        
    lot_size = risk_amount / (stop_loss_pips * pip_value * 100000)  # Standard lot
    
    return {
        "risk_amount": round(risk_amount, 2),
        "stop_loss_price_points": round(stop_loss_atr * current_atr, 5),
        "stop_loss_pips": round(stop_loss_pips, 1),
        "recommended_lots": round(lot_size, 2),
        "recommended_micro_lots": round(lot_size * 100, 0)
    }
