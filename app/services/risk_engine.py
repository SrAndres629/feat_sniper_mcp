import logging
import MetaTrader5 as mt5
from datetime import datetime, time as dtime
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.core.mt5_conn import mt5_conn

logger = logging.getLogger("MT5_Bridge.Services.Risk")

class RiskEngine:
    """
    Institutional Risk Management Engine.
    Handles adaptive lot sizing, drawdown protection, and exposure limits.
    """
    
    def __init__(self):
        self._daily_opening_balance: Optional[float] = None
        self._last_reset_date: Optional[str] = None

    async def _ensure_daily_balance(self):
        """Calcula o recupera el balance inicial del día para el drawdown."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            account_info = await mt5_conn.execute(mt5.account_info)
            if account_info:
                # En un sistema real, buscaríamos el balance al 00:00 en el historial
                # Aquí usamos el balance actual como base si es el primer inicio del día
                self._daily_opening_balance = account_info.balance
                self._last_reset_date = today
                logger.info(f"RiskEngine: Daily balance reset. Base: ${self._daily_opening_balance:.2f}")

    async def get_adaptive_lots(self, symbol: str, sl_points: int) -> float:
        """
        Calculates lot size based on account equity, risk percent, and SL distance.
        """
        if not settings.VOLATILITY_ADAPTIVE_LOTS:
            return 0.01

        account_info = await mt5_conn.execute(mt5.account_info)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)

        if not account_info or not symbol_info:
            return 0.01

        equity = account_info.equity
        cash_risk = equity * (settings.RISK_PER_TRADE_PERCENT / 100.0)
        
        point_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
        if sl_points <= 0: sl_points = 100

        calculated_lots = cash_risk / (sl_points * point_value)
        final_lots = round(max(symbol_info.volume_min, min(symbol_info.volume_max, calculated_lots)), 2)
        
        logger.info(f"RiskEngine: {final_lots} lots (Risk: ${cash_risk:.2f}, SL: {sl_points} pts)")
        return final_lots

    async def check_drawdown_limit(self) -> bool:
        """
        Vetoes trading if the daily drawdown limit is reached.
        Phantom Mode: Pauses execution but continues analysis.
        """
        await self._ensure_daily_balance()
        account_info = await mt5_conn.execute(mt5.account_info)
        if not account_info or not self._daily_opening_balance:
            return False

        real_loss = self._daily_opening_balance - account_info.equity
        current_drawdown = (real_loss / self._daily_opening_balance) * 100
        
        if current_drawdown > settings.MAX_DAILY_DRAWDOWN_PERCENT:
            logger.warning(f"⛔ PHANTOM MODE ACTIVE: Daily DD {current_drawdown:.2f}% (Limit: {settings.MAX_DAILY_DRAWDOWN_PERCENT}%)")
            logger.warning("Operativa pausada. Iniciando recalibración interna...")
            return False
        
        return True

    async def apply_trailing_stop(self, symbol: str, ticket: int, min_profit_pips: int = 10):
        """
        Aplica Trailing Stop basado en volatilidad (ATR).
        Si el precio se mueve 1.5 * ATR a favor, mueve el SL a Break Even.
        """
        pos = await mt5_conn.execute(mt5.positions_get, ticket=ticket)
        if not pos: return
        
        pos = pos[0]
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        
        # Obtener ATR para el cálculo dinámico
        from app.skills.market import get_volatility_metrics
        vol = await get_volatility_metrics(symbol)
        atr = vol.get("atr", 0)
        if atr <= 0: return

        trail_points = (atr * settings.ATR_TRAILING_MULTIPLIER) / symbol_info.point
        
        if pos.type == mt5.POSITION_TYPE_BUY:
            current_profit_points = (tick.bid - pos.price_open) / symbol_info.point
            # Lógica Breakeven (1.5 * ATR)
            if current_profit_points > (trail_points) and pos.sl < pos.price_open:
                logger.info(f"RiskEngine: Moving ticket {ticket} to BREAKEVEN (ATR Trail)")
                await self._modify_sl(ticket, pos.price_open, pos.tp)
            
            # Trailing dinámico
            new_sl = tick.bid - (trail_points * symbol_info.point)
            if new_sl > pos.sl and new_sl < tick.bid:
                await self._modify_sl(ticket, new_sl, pos.tp)
        
        elif pos.type == mt5.POSITION_TYPE_SELL:
            current_profit_points = (pos.price_open - tick.ask) / symbol_info.point
            # Lógica Breakeven (1.5 * ATR)
            if current_profit_points > (trail_points) and (pos.sl == 0 or pos.sl > pos.price_open):
                logger.info(f"RiskEngine: Moving ticket {ticket} to BREAKEVEN (ATR Trail)")
                await self._modify_sl(ticket, pos.price_open, pos.tp)
                
            # Trailing dinámico
            new_sl = tick.ask + (trail_points * symbol_info.point)
            if (pos.sl == 0 or new_sl < pos.sl) and new_sl > tick.ask:
                await self._modify_sl(ticket, new_sl, pos.tp)

    async def _modify_sl(self, ticket: int, sl: float, tp: float):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp
        }
        result = await mt5_conn.execute(mt5.order_send, request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"RiskEngine: Trailing SL updated for ticket {ticket} -> {sl}")

    # =========================================================================
    # TWIN-ENGINE HYBRID STRATEGY
    # =========================================================================
    
    async def get_capital_allocation(self) -> Dict[str, Any]:
        """
        Regla 50/50 Dinámica: Divide el margen libre para Scalp vs Swing.
        """
        account = await mt5_conn.execute(mt5.account_info)
        if not account:
            return {"scalp_capital": 0, "swing_capital": 0, "can_dual": False}
        
        free_margin = account.margin_free
        equity = account.equity
        
        # 50/50 Split
        scalp_capital = free_margin * 0.5
        swing_capital = free_margin * 0.5
        
        # Can we open 2 trades of 0.01?
        can_dual = await self.can_open_dual_trade("XAUUSD")
        
        return {
            "scalp_capital": round(scalp_capital, 2),
            "swing_capital": round(swing_capital, 2),
            "free_margin": round(free_margin, 2),
            "equity": round(equity, 2),
            "can_dual": can_dual,
            "max_positions": await self.max_positions_allowed()
        }
    
    async def can_open_dual_trade(self, symbol: str) -> bool:
        """
        Verifica si hay margen suficiente para 2 operaciones de 0.01.
        """
        account = await mt5_conn.execute(mt5.account_info)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        
        if not account or not symbol_info:
            return False
        
        # Margin required for 1 lot (then scale to 0.01)
        margin_per_lot = symbol_info.margin_initial if symbol_info.margin_initial > 0 else 1000
        margin_for_micro = (margin_per_lot / 100) * 0.01  # 0.01 lot
        
        # We need margin for 2 micro trades
        required_margin = margin_for_micro * 2
        
        return account.margin_free >= required_margin
    
    async def max_positions_allowed(self) -> int:
        """
        Growth Trigger: Determina cuántas posiciones podemos tener basado en equity.
        """
        account = await mt5_conn.execute(mt5.account_info)
        if not account:
            return 1
        
        equity = account.equity
        
        # Escala: $20=2pos, $50=3pos, $100=4pos
        if equity >= 100:
            return 4
        elif equity >= settings.EQUITY_UNLOCK_THRESHOLD:
            return 3  # Unlock 3rd position
        elif equity >= settings.INITIAL_CAPITAL:
            return 2  # Twin-Engine mode
        else:
            return 1  # Survival mode - Scalp only

risk_engine = RiskEngine()
