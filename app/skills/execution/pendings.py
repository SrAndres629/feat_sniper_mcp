from app.core.mt5_conn import mt5_conn, mt5
from app.models.schemas import TradeOrderRequest, ResponseModel, ErrorDetail
from .engine import send_order

async def place_limit_order(symbol: str, action: str, volume: float, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> ResponseModel:
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    if not tick: return ResponseModel(status="error", error=ErrorDetail(code="NO_TICK", message="No market data"))
    curr = tick.ask if action == "BUY" else tick.bid
    if action == "BUY" and price >= curr: return ResponseModel(status="error", error=ErrorDetail(code="INVALID_LIMIT", message="Buy Limit must be BELOW current price"))
    if action == "SELL" and price <= curr: return ResponseModel(status="error", error=ErrorDetail(code="INVALID_LIMIT", message="Sell Limit must be ABOVE current price"))
    return await send_order(TradeOrderRequest(symbol=symbol, action=f"{action}_LIMIT", volume=volume, price=price, sl=sl, tp=tp, comment=comment), urgency_score=0.3)

async def place_stop_order(symbol: str, action: str, volume: float, price: float, sl: float = 0.0, tp: float = 0.0, comment: str = "") -> ResponseModel:
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    if not tick: return ResponseModel(status="error", error=ErrorDetail(code="NO_TICK", message="No market data"))
    curr = tick.ask if action == "BUY" else tick.bid
    if action == "BUY" and price <= curr: return ResponseModel(status="error", error=ErrorDetail(code="INVALID_STOP", message="Buy Stop must be ABOVE current price"))
    if action == "SELL" and price >= curr: return ResponseModel(status="error", error=ErrorDetail(code="INVALID_STOP", message="Sell Stop must be BELOW current price"))
    return await send_order(TradeOrderRequest(symbol=symbol, action=f"{action}_STOP", volume=volume, price=price, sl=sl, tp=tp, comment=comment), urgency_score=0.7)
