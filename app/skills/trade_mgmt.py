import logging
from typing import Dict, Any, Optional
try:
    import MetaTrader5 as mt5
except ImportError:
    from unittest.mock import MagicMock
    mt5 = MagicMock()
from app.core.mt5_conn import mt5_conn
from app.models.schemas import PositionManageRequest, ResponseModel, PositionActionResponse, ErrorDetail

logger = logging.getLogger("MT5_Bridge.Skills.TradeMgmt")

async def manage_position(data: PositionManageRequest) -> ResponseModel[PositionActionResponse]:
    """
    Gestiona posiciones abiertas: Cierre total/parcial o modificacin de SL/TP.
    """
    ticket = data.ticket
    
    # 1. Obtener info de la posicin
    positions = await mt5_conn.execute(mt5.positions_get, ticket=ticket)
    if not positions or len(positions) == 0:
        return ResponseModel(
            status="error",
            error=ErrorDetail(
                code="POSITION_NOT_FOUND",
                message=f"No se encontr la posicin con ticket {ticket}.",
                suggestion="Verifica que el ticket sea correcto y que la posicin siga abierta."
            )
        )
    
    pos = positions[0]
    symbol = pos.symbol
    
    # 2. Lgica de CIERRE / ELIMINACIN
    if data.action == "CLOSE" or data.action == "DELETE":
        # DELETE es para rdenes pendientes
        if data.action == "DELETE":
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            result = await mt5_conn.execute(mt5.order_send, request)
        else:
            # Lgica de CLOSE (Dealer execution)
            volume_to_close = data.volume if data.volume else pos.volume
            # Validar volumen de cierre
            if volume_to_close > pos.volume:
                return ResponseModel(
                    status="error",
                    error=ErrorDetail(
                        code="INVALID_CLOSE_VOLUME",
                        message=f"Intentando cerrar {volume_to_close} pero la posicin solo tiene {pos.volume}.",
                        suggestion="Ajusta el volumen a cerrar para que sea menor o igual al actual."
                    )
                )
            tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume_to_close,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "n8n-Auto-Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = await mt5_conn.execute(mt5.order_send, request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code=f"ACTION_FAILED_{result.retcode}",
                    message=f"Fallo al ejecutar {data.action}: {result.comment}",
                    suggestion="Reintenta o verifica el ticket."
                )
            )
        
        return ResponseModel(
            status="success",
            data=PositionActionResponse(
                ticket=ticket,
                status="EXECUTED",
                message=f"Ticket {ticket} procesado con xito ({data.action})."
            )
        )

    # 3. Lgica de MODIFICACIN (SL/TP/PRICE)
    elif data.action == "MODIFY":
        # Para rdenes pendientes el ticket es 'order', para posiciones es 'position'
        # Pero MT5 suele discriminar por el campo 'action'
        request = {
            "action": mt5.TRADE_ACTION_SLTP if not data.price else mt5.TRADE_ACTION_MODIFY,
            "symbol": symbol,
            "sl": data.sl if data.sl is not None else pos.sl,
            "tp": data.tp if data.tp is not None else pos.tp,
        }
        
        if not data.price:
            request["position"] = ticket
        else:
            request["order"] = ticket
            request["price"] = data.price
        
        result = await mt5_conn.execute(mt5.order_send, request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code=f"MODIFY_FAILED_{result.retcode}",
                    message=f"Fallo al modificar: {result.comment}",
                    suggestion="Verifica que el SL/TP no est demasiado cerca del precio (Stops Level)."
                )
            )
            
        return ResponseModel(
            status="success",
            data=PositionActionResponse(
                ticket=ticket,
                status="MODIFIED",
                message=f"SL/TP de la posicin {ticket} actualizados."
            )
        )

    return ResponseModel(status="error", error=ErrorDetail(code="INVALID_ACTION", message="Accin no soportada.", suggestion="Usa CLOSE o MODIFY."))

async def apply_neuro_trailing(ticket: int, neural_sl_price: float, volatility_regime: str) -> ResponseModel[PositionActionResponse]:
    """
    Apply Dynamic Trailing Stop based on Neural Volatility Regime.
    
    Args:
        ticket: Ticket ID.
        neural_sl_price: Proposed SL price by Neural Net.
        volatility_regime: 'HIGH', 'NORMAL', 'LOW' (affects buffer).
    """
    # 1. Get Position
    positions = await mt5_conn.execute(mt5.positions_get, ticket=ticket)
    if not positions:
        return ResponseModel(status="error", error=ErrorDetail(code="POS_NOT_FOUND", message="Position closing or closed."))
    pos = positions[0]
    
    # 2. Validate Direction
    is_buy = pos.type == mt5.ORDER_TYPE_BUY
    current_sl = pos.sl
    
    # Logic: Only move SL in favor of trade (Ratchet)
    if is_buy:
        if neural_sl_price <= current_sl:
            return ResponseModel(status="success", data=PositionActionResponse(ticket=ticket, status="SKIPPED", message="New SL < Current SL ( Buy)."))
    else:
        if neural_sl_price >= current_sl and current_sl > 0:
            return ResponseModel(status="success", data=PositionActionResponse(ticket=ticket, status="SKIPPED", message="New SL > Current SL (Sell)."))

    # 3. Apply Volatility Buffer (Guardrails)
    # If High Volatility, ensure we are not suffocation the trade.
    symbol_info = await mt5_conn.execute(mt5.symbol_info, pos.symbol)
    point = symbol_info.point
    min_dist = 50 * point # 5 pips minimum distance from price
    
    current_price = await mt5_conn.execute(mt5.symbol_info_tick, pos.symbol)
    price = current_price.bid if is_buy else current_price.ask
    
    dist_to_price = abs(price - neural_sl_price)
    
    if dist_to_price < min_dist:
        logger.warning(f"NeuroTrailing: SL too close to price ({dist_to_price/point} pips). Buffering.")
        # Push back
        if is_buy: neural_sl_price = price - min_dist
        else: neural_sl_price = price + min_dist

    # 4. Execute Modification
    return await manage_position(PositionManageRequest(
        ticket=ticket, 
        action="MODIFY", 
        sl=neural_sl_price,
        symbol=pos.symbol
    ))
