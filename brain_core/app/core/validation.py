import logging
import MetaTrader5 as mt5
from typing import Optional, Tuple
from app.core.mt5_conn import mt5_conn
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Core.Validation")

class OrderValidator:
    """
    Motor de validacin inteligente para rdenes de trading y gestin de riesgo.
    """

    @staticmethod
    async def validate_order(symbol: str, volume: float, action: str, price: Optional[float] = None, sl: Optional[float] = None, tp: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Valida que una orden cumpla con los requisitos del broker y las reglas de riesgo.
        """
        # 1. Obtener info del smbolo
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        if not symbol_info:
            return False, f"Smbolo {symbol} no encontrado."

        if not symbol_info.visible:
            await mt5_conn.execute(mt5.symbol_select, symbol, True)

        # 2. Validar Volumen (Lots)
        if volume < symbol_info.volume_min:
            return False, f"Volumen {volume} inferior al mnimo permitido ({symbol_info.volume_min})."
        
        if volume > symbol_info.volume_max:
            return False, f"Volumen {volume} superior al mximo permitido ({symbol_info.volume_max})."
        
        # Validar Step (Paso de volumen) con tolerancia a punto flotante
        # Algoritmo: volume debe ser mltiplo de step
        # round(volume / step) * step debera ser casi igual a volume
        steps = round(volume / symbol_info.volume_step)
        expected_vol = steps * symbol_info.volume_step
        
        if abs(volume - expected_vol) > 0.00001:
             return False, f"Volumen {volume} no cumple con el paso de volumen ({symbol_info.volume_step})."

        # 3. Validar Stops (SL/TP) y Congelacin (Freeze Level/Stop Level)
        # Obtenemos el precio actual
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        if not tick:
            return False, "No se pudo obtener el precio para validar stops."

        # Referencia para validacin depende del tipo de orden
        ref_price = price if price else (tick.ask if "BUY" in action else tick.bid)
        
        stop_level_points = symbol_info.stops_level
        point_size = symbol_info.point
        min_distance = stop_level_points * point_size

        if sl:
            distance_sl = abs(ref_price - sl)
            if distance_sl < min_distance:
                return False, f"Stop Loss demasiado cerca. Distancia mnima: {stop_level_points} puntos ({min_distance})."

        if tp:
            distance_tp = abs(ref_price - tp)
            if distance_tp < min_distance:
                return False, f"Take Profit demasiado cerca. Distancia mnima: {stop_level_points} puntos ({min_distance})."

        # 4. Validacin de Riesgo (Drawdown)
        account_info = await mt5_conn.execute(mt5.account_info)
        if not account_info:
            return False, "No se pudo obtener la informacin de la cuenta para validar riesgo."

        # Drawdown actual
        daily_loss_percent = ((account_info.balance - account_info.equity) / account_info.balance) * 100 if account_info.balance > 0 else 0
        if daily_loss_percent > settings.MAX_DAILY_DRAWDOWN_PERCENT:
            return False, f"Lmite de Drawdown Diario excedido ({daily_loss_percent:.2f}% > {settings.MAX_DAILY_DRAWDOWN_PERCENT}%). Trading bloqueado."

        # 5. Institutional Constraints (New)
        positions_total = await mt5_conn.execute(mt5.positions_total)
        if positions_total >= settings.MAX_OPEN_POSITIONS:
            return False, f"Lmite institucional de posiciones alcanzado ({positions_total}/{settings.MAX_OPEN_POSITIONS})."

        # Check for Correlation Exposure (Simplistic approach: how many trades in same currency group)
        # Institutional standard: Check Correlation Matrix here. 
        # For now, we limit total trades per symbol to prevent over-concentration.
        symbol_positions = 0
        all_positions = await mt5_conn.execute(mt5.positions_get)
        if all_positions:
            for p in all_positions:
                if p.symbol == symbol:
                    symbol_positions += 1
        
        if symbol_positions >= 3: # Max 3 trades per symbol
            return False, f"Sobre-exposicin detectada en {symbol}. Mximo 3 posiciones permitidas."

        return True, None

    @staticmethod
    async def validate_margin(symbol: str, volume: float, action: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica si hay margen suficiente antes de enviar la orden.
        """
        order_type = mt5.ORDER_TYPE_BUY if "BUY" in action else mt5.ORDER_TYPE_SELL
        margin_required = await mt5_conn.execute(mt5.order_calc_margin, order_type, symbol, volume, 0.0) # 0.0 es el precio actual
        
        account_info = await mt5_conn.execute(mt5.account_info)
        if account_info.margin_free < margin_required:
            return False, f"Margen insuficiente. Requerido: {margin_required}, Disponible: {account_info.margin_free}"
        
        return True, None
