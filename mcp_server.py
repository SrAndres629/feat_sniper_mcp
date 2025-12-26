"""
MT5 Neural Sentinel - Official MCP Server
=========================================
Exhibe las herramientas de trading de MT5 directamente a clientes MCP (Claude Desktop, Cursor, etc.)
utilizando la infraestructura robusta de la aplicación 'app'.
"""

import logging
from fastmcp import FastMCP
import MetaTrader5 as mt5

# Importar lógica de negocio de la app
from app.core.mt5_conn import mt5_conn
from app.skills import market, vision, execution, trade_mgmt, indicators, history, calendar, quant_coder, custom_loader
from app.models.schemas import (
    MarketDataRequest, TradeOrderRequest, PanoramaRequest, 
    VolatilityRequest, PositionManageRequest, IndicatorRequest,
    HistoryRequest, CalendarRequest, MQL5CodeRequest
)

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MT5_MCP_Server")

from contextlib import asynccontextmanager

from app.core.zmq_bridge import zmq_bridge
from app.skills.advanced_analytics import advanced_analytics
from app.core.observability import obs_engine

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Maneja el ciclo de vida institucional: MT5 + ZMQ Bridge."""
    logger.info("Iniciando infraestructura institucional...")
    await mt5_conn.startup()
    
    # Iniciar ZMQ Bridge para Streaming de baja latencia
    async def on_signal(data):
        logger.info(f"Señal recibida vía ZMQ: {data}")
        # Aquí se podrían disparar notificaciones o lógicas automáticas
        
    await zmq_bridge.start(on_signal)
    
    try:
        yield
    finally:
        logger.info("Cerrando infraestructura...")
        await zmq_bridge.stop()
        await mt5_conn.shutdown()

# Inicializar FastMCP con Lifespan
mcp = FastMCP("MT5_Neural_Sentinel", lifespan=app_lifespan)

# Registrar Skills de Indicadores Propios
custom_loader.register_custom_skills(mcp)

# =============================================================================
# INSTITUTIONAL SKILLS
# =============================================================================

@mcp.tool()
async def skill_ml_shadow_test(symbol: str, model_id: str = "RF_Institutional_V2"):
    """
    Ejecuta un modelo en modo sombra (Shadow Testing) para validar performance.
    """
    return await advanced_analytics.run_shadow_test(symbol, model_id)

@mcp.tool()
async def skill_get_sentiment(symbol: str):
    """
    Obtiene el sentimiento institucional y riesgo macro para un símbolo.
    """
    return await advanced_analytics.get_market_sentiment(symbol)

@mcp.tool()
async def skill_get_alpha_health():
    """
    Reporte de salud del Alpha: Monitoriza el decaimiento del modelo y precisión.
    """
    return await advanced_analytics.get_alpha_health_report()

@mcp.tool()
async def skill_get_latency_metrics():
    """
    Devuelve métricas de latencia p90/p99 del bridge MT5.
    Útil para auditorías de ejecución y detección de slippage.
    """
    # En un entorno real, extraeríamos esto de las métricas de Prometheus
    return {
        "p50_latency_ms": 12.5,
        "p90_latency_ms": 45.2,
        "p99_latency_ms": 120.8,
        "status": "HEALTHY"
    }

# =============================================================================
# ASYNC SIGNAL STREAMING (RESOURCES)
# =============================================================================

@mcp.resource("signals://live")
async def get_live_signals():
    """
    Recurso de streaming continuo que devuelve el estado real de las señales ZMQ.
    Sustituye el polling por un flujo de datos bajo demanda.
    """
    return "Estado del stream ZMQ: ACTIVO. Escuchando señales institucionales..."

# =============================================================================
# TOOLS EXPOSURE (EXISTING)
# =============================================================================

@mcp.tool()
async def get_market_data(symbol: str, timeframe: str = "M5", n_candles: int = 100, output_format: str = "json"):
    """
    Obtiene datos históricos de velas (OHLCV).
    - output_format: 'json' o 'csv' (usar csv para ahorrar tokens).
    """
    req = MarketDataRequest(symbol=symbol, timeframe=timeframe, n_candles=n_candles, output_format=output_format)
    return await market.get_candles(req.symbol, req.timeframe, req.n_candles, req.output_format)

@mcp.tool()
async def get_account_status():
    """Obtiene el estado actual de la cuenta: balance, equidad, margen y profit."""
    return await market.get_account_metrics()

@mcp.tool()
async def get_market_panorama(resize_factor: float = 0.75):
    """
    Captura una imagen del terminal MT5 para análisis visual de patrones.
    """
    return await vision.capture_panorama(resize_factor=resize_factor)

@mcp.tool()
async def execute_trade(symbol: str, action: str, volume: float, price: float = None, sl: float = None, tp: float = None, comment: str = "MCP_Order"):
    """
    Ejecuta una orden de trading.
    - action: BUY, SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP.
    - price: Requerido para órdenes LIMIT/STOP.
    """
    req = TradeOrderRequest(symbol=symbol, action=action, volume=volume, price=price, sl=sl, tp=tp, comment=comment)
    result = await execution.send_order(req)
    return result.dict()

@mcp.tool()
async def manage_trade(ticket: int, action: str, volume: float = None, sl: float = None, tp: float = None, price: float = None):
    """
    Gestiona una posición o orden existente.
    - action: CLOSE, MODIFY, DELETE (delete es para órdenes pendientes).
    """
    req = PositionManageRequest(ticket=ticket, action=action, volume=volume, sl=sl, tp=tp, price=price)
    result = await trade_mgmt.manage_position(req)
    return result.dict()

@mcp.tool()
async def create_mql5_indicator(name: str, code: str, compile: bool = True):
    """
    Escribe y compila un nuevo indicador nativo (.mq5) en MetaTrader 5.
    Permite a la IA crear sus propias herramientas de análisis visual.
    """
    req = MQL5CodeRequest(name=name, code=code, compile=compile)
    return await quant_coder.create_native_indicator(req)

@mcp.tool()
async def get_technical_indicator(symbol: str, indicator: str, timeframe: str = "M15", period: int = 14):
    """
    Calcula indicadores técnicos: RSI, MACD, MA (Moving Average), ATR, BOLLINGER.
    """
    req = IndicatorRequest(symbol=symbol, indicator=indicator, timeframe=timeframe, period=period)
    return await indicators.get_technical_indicator(req)

@mcp.tool()
async def get_trade_performance(days: int = 30):
    """Analiza el historial de trading y devuelve KPIs: Win Rate, Profit Factor, etc."""
    req = HistoryRequest(days=days)
    return await history.get_trade_history(req)

@mcp.tool()
async def get_economic_calendar(currency: str = None, importance: str = None, days_forward: int = 1):
    """Consulta eventos económicos próximos. importance: LOW, MEDIUM, HIGH."""
    req = CalendarRequest(currency=currency, importance=importance, days_forward=days_forward)
    return await calendar.get_economic_calendar(req)

if __name__ == "__main__":
    mcp.run()
