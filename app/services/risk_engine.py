import logging
import MetaTrader5 as mt5
from typing import Dict, Any, Optional
from app.core.config import settings
from app.core.mt5_conn import mt5_conn

logger = logging.getLogger("MT5_Bridge.Services.Risk")

class RiskEngine:
    """
    Institutional Risk Management Engine.
    Handles adaptive lot sizing, drawdown protection, and exposure limits.
    """

    @staticmethod
    async def get_adaptive_lots(symbol: str, sl_points: int) -> float:
        """
        Calculates lot size based on account equity, risk percent, and SL distance.
        Uses volatility adjustment if enabled.
        """
        if not settings.VOLATILITY_ADAPTIVE_LOTS:
            return 0.01 # Minimum safety default

        account_info = await mt5_conn.execute(mt5.account_info)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)

        if not account_info or not symbol_info:
            logger.error(f"Failed to fetch data for lot calculation on {symbol}")
            return 0.01

        # Calculate cash at risk
        equity = account_info.equity
        cash_risk = equity * (settings.RISK_PER_TRADE_PERCENT / 100.0)

        # Calculate tick value/point value
        # formula: lots = cash_risk / (sl_points * tick_value_per_lot)
        # Note: simplistic for demonstration, institutional requires currency conversion
        
        point_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
        if sl_points <= 0:
            sl_points = 100 # Default fallback to avoid div zero

        calculated_lots = cash_risk / (sl_points * point_value)
        
        # Clamp between min/max lots
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        
        final_lots = round(max(min_lot, min(max_lot, calculated_lots)), 2)
        
        logger.info(f"RiskEngine: Calculated {final_lots} lots for {symbol} (Risk: ${cash_risk:.2f}, SL: {sl_points} pts)")
        return final_lots

    @staticmethod
    async def check_drawdown_limit() -> bool:
        """
        Vetoes trading if the daily drawdown limit is reached.
        """
        account_info = await mt5_conn.execute(mt5.account_info)
        if not account_info:
            return False

        current_drawdown = ((account_info.balance - account_info.equity) / account_info.balance) * 100
        if current_drawdown > settings.MAX_DAILY_DRAWDOWN_PERCENT:
            logger.warning(f"CRITICAL: Daily drawdown {current_drawdown:.2f}% exceeds limit!")
            return False
        
        return True

    @staticmethod
    async def get_total_exposure() -> float:
        """
        Calculates total margin used vs total equity.
        """
        account_info = await mt5_conn.execute(mt5.account_info)
        if not account_info or account_info.margin == 0:
            return 0.0
        
        return (account_info.margin / account_info.equity) * 100

risk_engine = RiskEngine()
