import logging
from typing import Optional, Tuple
from app.core.mt5_conn import mt5_conn, mt5
from app.core.config import settings

logger = logging.getLogger("SystemGuard.Validator")

class OrderValidator:
    @staticmethod
    async def validate_order(symbol: str, volume: float, action: str, price: float = None, sl: float = None, tp: float = None) -> Tuple[bool, Optional[str]]:
        si = await mt5_conn.execute(mt5.symbol_info, symbol)
        if not si: return False, f"Symbol {symbol} not found"
        if not si.visible: await mt5_conn.execute(mt5.symbol_select, symbol, True)
        if volume < si.volume_min: return False, f"Vol {volume} < min {si.volume_min}"
        if volume > si.volume_max: return False, f"Vol {volume} > max {si.volume_max}"
        if abs(volume - round(volume/si.volume_step)*si.volume_step) > 1e-5: return False, f"Vol {volume} invalid step {si.volume_step}"
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        if not tick: return False, "No price data"
        ref = price or (tick.ask if "BUY" in action else tick.bid)
        md = si.stops_level * si.point
        if sl and abs(ref - sl) < md: return False, f"SL too close ({md} pts req)"
        if tp and abs(ref - tp) < md: return False, f"TP too close ({md} pts req)"
        acc = await mt5_conn.execute(mt5.account_info)
        if acc:
            dl = ((acc.balance - acc.equity) / acc.balance * 100) if acc.balance > 0 else 0
            if dl > settings.MAX_DAILY_DRAWDOWN_PERCENT: return False, f"Drawdown {dl:.2f}% > limit"
        pos = await mt5_conn.execute(mt5.positions_total)
        if pos >= settings.MAX_OPEN_POSITIONS: return False, f"Pos limit {settings.MAX_OPEN_POSITIONS} hit"
        return True, None

    @staticmethod
    async def validate_margin(symbol, volume, action) -> Tuple[bool, Optional[str]]:
        ot = mt5.ORDER_TYPE_BUY if "BUY" in action else mt5.ORDER_TYPE_SELL
        mr = await mt5_conn.execute(mt5.order_calc_margin, ot, symbol, volume, 0.0)
        acc = await mt5_conn.execute(mt5.account_info)
        if not acc or acc.margin_free < mr: return False, "Insufficient margin"
        return True, None
