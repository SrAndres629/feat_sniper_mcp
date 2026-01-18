import logging
from typing import Dict, Any
from app.core.mt5_conn import mt5_conn, mt5
from app.core.config import settings

logger = logging.getLogger("Execution.Twin")

async def execute_twin_trade(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Protocolo Twin-Entry: Abre 2 órdenes simultáneas con diferentes objetivos.
    """
    from app.services.risk import risk_engine
    symbol = signal.get("symbol", "XAUUSD")
    direction = signal.get("direction", "BUY").upper()
    
    alloc = await risk_engine.get_capital_allocation()
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    s_info = await mt5_conn.execute(mt5.symbol_info, symbol)
    
    if not tick or not s_info: return {"status": "error", "message": "Market data error"}
    
    # targets
    val = s_info.trade_tick_value
    scalp_pips = int(settings.SCALP_TARGET_USD / (val * 0.01)) if val > 0 else 200
    swing_pips = int(settings.SWING_TARGET_USD / (val * 0.01)) if val > 0 else 1000
    sl_pips = 20
    
    price = tick.ask if direction == "BUY" else tick.bid
    sl = price - (sl_pips * s_info.point) if direction == "BUY" else price + (sl_pips * s_info.point)
    s_tp = price + (scalp_pips * s_info.point) if direction == "BUY" else price - (scalp_pips * s_info.point)
    w_tp = price + (swing_pips * s_info.point) if direction == "BUY" else price - (swing_pips * s_info.point)

    res = {"mode": "TWIN" if alloc["can_dual"] else "SCALP_ONLY"}
    
    # Scalp
    vol = await risk_engine.calculate_dynamic_lot(0.95, 0.01, symbol, sl_pips)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": vol, "type": mt5.ORDER_TYPE_BUY if direction=="BUY" else mt5.ORDER_TYPE_SELL,
           "price": price, "sl": sl, "tp": s_tp, "deviation": 20, "magic": settings.MAGIC_SCALP, "comment": "TWIN_SCALP", "type_filling": mt5.ORDER_FILLING_RETURN}
    
    s_res = await mt5_conn.execute(mt5.order_send, req)
    if s_res and s_res.retcode == mt5.TRADE_RETCODE_DONE: res["scalp_ticket"] = s_res.order
    
    # Swing
    if alloc["can_dual"]:
        svol = await risk_engine.calculate_dynamic_lot(0.85, 0.01, symbol, sl_pips)
        sreq = {**req, "volume": svol, "tp": w_tp, "magic": settings.MAGIC_SWING, "comment": "TWIN_SWING"}
        w_res = await mt5_conn.execute(mt5.order_send, sreq)
        if w_res and w_res.retcode == mt5.TRADE_RETCODE_DONE: res["swing_ticket"] = w_res.order
        
    res["status"] = "success" if res.get("scalp_ticket") else "failed"
    return res
