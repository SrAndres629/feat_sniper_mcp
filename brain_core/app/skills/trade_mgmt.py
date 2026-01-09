import logging
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
from app.core.mt5_conn import mt5_conn
from app.models.schemas import PositionManageRequest, ResponseModel, PositionActionResponse, ErrorDetail

logger = logging.getLogger("MT5_Bridge.Skills.TradeMgmt")

async def manage_position(data: PositionManageRequest) -> ResponseModel[PositionActionResponse]:
    """
    Gestiona posiciones abiertas: Cierre total/parcial o modificación de SL/TP.
    """
    ticket = data.ticket
    
    # 1. Obtener info de la posición
    positions = await mt5_conn.execute(mt5.positions_get, ticket=ticket)
    if not positions or len(positions) == 0:
        return ResponseModel(
            status="error",
            error=ErrorDetail(
                code="POSITION_NOT_FOUND",
                message=f"No se encontró la posición con ticket {ticket}.",
                suggestion="Verifica que el ticket sea correcto y que la posición siga abierta."
            )
        )
    
    pos = positions[0]
    symbol = pos.symbol
    
    # 2. Lógica de CIERRE / ELIMINACIÓN
    if data.action == "CLOSE" or data.action == "DELETE":
        # DELETE es para órdenes pendientes
        if data.action == "DELETE":
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            result = await mt5_conn.execute(mt5.order_send, request)
        else:
            # Lógica de CLOSE (Dealer execution)
            volume_to_close = data.volume if data.volume else pos.volume
            # Validar volumen de cierre
            if volume_to_close > pos.volume:
                return ResponseModel(
                    status="error",
                    error=ErrorDetail(
                        code="INVALID_CLOSE_VOLUME",
                        message=f"Intentando cerrar {volume_to_close} pero la posición solo tiene {pos.volume}.",
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
                message=f"Ticket {ticket} procesado con éxito ({data.action})."
            )
        )

    # 3. Lógica de MODIFICACIÓN (SL/TP/PRICE)
    elif data.action == "MODIFY":
        # Para órdenes pendientes el ticket es 'order', para posiciones es 'position'
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
                    suggestion="Verifica que el SL/TP no esté demasiado cerca del precio (Stops Level)."
                )
            )
            
        return ResponseModel(
            status="success",
            data=PositionActionResponse(
                ticket=ticket,
                status="MODIFIED",
                message=f"SL/TP de la posición {ticket} actualizados."
            )
        )

    return ResponseModel(status="error", error=ErrorDetail(code="INVALID_ACTION", message="Acción no soportada.", suggestion="Usa CLOSE o MODIFY."))
