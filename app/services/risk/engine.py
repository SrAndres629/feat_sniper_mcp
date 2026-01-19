import logging
import os
from typing import Dict, Any, Optional
from app.core.config import settings
from app.core.mt5_conn import mt5_conn, mt5
from .vault import TheVault
from .logic import calculate_damped_kelly, get_neural_allocation, DrawdownMonitor, check_trading_veto

logger = logging.getLogger("Risk.Engine")

class RiskEngine:
    def __init__(self):
        self.vault = TheVault(initial_capital=settings.INITIAL_CAPITAL)
        self.drawdown = DrawdownMonitor()

    async def calculate_dynamic_lot(self, confidence: float, vol: float, symbol: str, sl: int = 200, m_data: Dict = {}) -> float:
        if confidence < 0.60: return 0.01
        from app.services.circuit_breaker import circuit_breaker
        cb_mult = await circuit_breaker.get_lot_multiplier()
        ok, res = await check_trading_veto(symbol, m_data, cb_mult)
        if not ok: return 0.0
        r_mult = 0.5 if res == "TURBULENT" else 1.0
        target_f = calculate_damped_kelly(confidence, m_data.get("brain_uncertainty", 0.05))
        n_mult = target_f / (settings.effective_risk_cap / 100.0 or 0.01)
        base_lot = await self.get_adaptive_lots(symbol, sl, n_mult)
        final = base_lot * cb_mult * r_mult
        return max(0.01, final) if final > 0 else 0.0

    async def get_adaptive_lots(self, symbol: str, sl: int, n_mult: float = 1.0) -> float:
        if n_mult <= 0 or sl <= 0: return 0.01
        acc, si = await mt5_conn.execute(mt5.account_info), await mt5_conn.execute(mt5.symbol_info, symbol)
        if not acc or not si: return 0.01
        pval = si.trade_tick_value / si.trade_tick_size * si.point or 10
        risk_usd = acc.equity * (settings.effective_risk_cap / 100) * n_mult
        lots = risk_usd / (sl * pval)
        if lots < si.volume_min and (si.volume_min * sl * pval) <= (acc.equity * (settings.effective_risk_cap / 100)):
            lots = si.volume_min
        lots = max(si.volume_min, min(lots, si.volume_max))
        return round(round(lots / si.volume_step) * si.volume_step, 2)

    async def apply_trailing_stop(self, symbol: str, ticket: int):
        pos = await mt5_conn.execute(mt5.positions_get, ticket=ticket)
        if not pos: return
        p, si, t = pos[0], await mt5_conn.execute(mt5.symbol_info, symbol), await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        from app.skills.market import get_volatility_metrics
        v = await get_volatility_metrics(symbol)
        atr = v.get("atr", 0)
        if atr <= 0: return
        tp = (atr * settings.ATR_TRAILING_MULTIPLIER) / si.point
        if p.type == mt5.POSITION_TYPE_BUY:
            if (t.bid - p.price_open) / si.point > tp and p.sl < p.price_open: await self._mod(ticket, p.price_open, p.tp)
            ns = t.bid - (tp * si.point)
            if ns > p.sl and ns < t.bid: await self._mod(ticket, ns, p.tp)
        elif p.type == mt5.POSITION_TYPE_SELL:
            if (p.price_open - t.ask) / si.point > tp and (p.sl == 0 or p.sl > p.price_open): await self._mod(ticket, p.price_open, p.tp)
            ns = t.ask + (tp * si.point)
            if (p.sl == 0 or ns < p.sl) and ns > t.ask: await self._mod(ticket, ns, p.tp)

    async def _mod(self, tkt, sl, tp):
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": tkt, "sl": sl, "tp": tp}
        await mt5_conn.execute(mt5.order_send, req)

    async def check_trading_veto(self, symbol: str, action: str, price: float) -> Dict[str, Any]:
        """Wrapper for logic.check_trading_veto"""
        # Mocking context for smoke test
        context = {"symbol": symbol, "action": action, "price": price} 
        cb_mult = 1.0
        ok, status = await check_trading_veto(symbol, context, cb_mult)
        return {"status": status, "allowed": ok}
