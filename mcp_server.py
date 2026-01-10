import logging
import sys
from fastmcp import FastMCP
from contextlib import asynccontextmanager

# Configuración de Logging CRÍTICA para MCP
# Todo log debe ir a stderr para no romper el protocolo JSON-RPC en stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("MT5_MCP_Server")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logger.warning("MetaTrader5 module not found found. Running in Linux/Docker mode? Some local-only features will be disabled.")


# Importar lógica de negocio de la app
from app.core.mt5_conn import mt5_conn
from app.skills import (
    market, vision, execution, trade_mgmt, 
    indicators, history, calendar, 
    quant_coder, custom_loader,
    tester, unified_model, remote_compute # Nuevos módulos
)
from app.models.schemas import (
    MarketDataRequest, TradeOrderRequest, PositionManageRequest, 
    IndicatorRequest, HistoryRequest, CalendarRequest, MQL5CodeRequest
)

from app.core.zmq_bridge import zmq_bridge
from app.skills.advanced_analytics import advanced_analytics
from app.services.supabase_sync import supabase_sync

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Maneja el ciclo de vida institucional: MT5 + ZMQ Bridge."""
    logger.info("Iniciando infraestructura institucional...")
    await mt5_conn.startup()
    
    # Iniciar ZMQ Bridge para Streaming de baja latencia
    async def on_signal(data):
        logger.info(f"Señal recibida vía ZMQ: {data}")
        # Sincronizar con la nube de forma asíncrona (Fire & Forget)
        import asyncio
        asyncio.create_task(supabase_sync.log_signal(data))
        
    await zmq_bridge.start(on_signal)
    
    try:
        yield
    finally:
        logger.info("Cerrando infraestructura...")
        await zmq_bridge.stop()
        await mt5_conn.shutdown()

# Inicializar FastMCP
mcp = FastMCP("MT5_Neural_Sentinel", lifespan=app_lifespan)

# Registrar Skills de Indicadores Propios
custom_loader.register_custom_skills(mcp)

# Registrar Skills de Cómputo Remoto (NEXUS)
remote_compute.register_remote_skills(mcp)

# =============================================================================
# PILLAR 1: CÓRTEX DE COMPILACIÓN (Code Builder)
# =============================================================================

@mcp.tool()
async def create_and_compile_indicator(name: str, code: str, compile: bool = True):
    """
    Escribe y compila un nuevo indicador .mq5. 
    Retorna el log de compilación para auto-corrección.
    """
    req = MQL5CodeRequest(name=name, code=code, compile=compile)
    return await quant_coder.create_native_indicator(req)

# =============================================================================
# PILLAR 2: OJO DE HALCÓN (Data Bridge)
# =============================================================================

@mcp.tool()
async def get_market_snapshot(symbol: str, timeframe: str = "M5"):
    """
    Radiografía completa del mercado: Precio, Vela actual, Volatilidad y Cuenta.
    Ideal para decisiones de alta frecuencia.
    """
    return await market.get_market_snapshot(symbol, timeframe)

@mcp.tool()
async def get_candles(symbol: str, timeframe: str = "M5", n_candles: int = 100):
    """Obtiene velas históricas (OHLCV)."""
    return await market.get_candles(symbol, timeframe, n_candles)

@mcp.tool()
async def get_market_panorama(resize_factor: float = 0.75):
    """Captura visual del gráfico."""
    return await vision.capture_panorama(resize_factor=resize_factor)

# =============================================================================
# PILLAR 3: SIMULADOR DE SOMBRAS (Backtest Automator)
# =============================================================================

@mcp.tool()
async def run_shadow_backtest(
    expert_name: str, 
    symbol: str, 
    period: str = "H1", 
    days: int = 30,
    deposit: int = 10000
):
    """
    Ejecuta una prueba de estrategia en segundo plano usando el Strategy Tester de MT5.
    """
    # Calcular fechas
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    return await tester.run_strategy_test(
        expert_name=expert_name,
        symbol=symbol,
        period=period,
        date_from=start_date.strftime("%Y.%m.%d"),
        date_to=end_date.strftime("%Y.%m.%d"),
        deposit=deposit
    )

# =============================================================================
# PILLAR 4: INYECCIÓN ESTRATÉGICA (Unified Model)
# =============================================================================

@mcp.tool()
async def query_unified_brain(sql_query: str):
    """
    Consulta la base de datos de conocimiento unificado (Unified Model).
    Solo permite SELECT.
    """
    return await unified_model.unified_db.query_custom_sql(sql_query)

# =============================================================================
# EXECUTION & MANAGEMENT SKILLS
# =============================================================================

@mcp.tool()
async def execute_trade(symbol: str, action: str, volume: float, price: float = None, sl: float = None, tp: float = None, comment: str = "MCP_Order"):
    """Ejecuta órdenes: BUY, SELL, LIMIT, STOP."""
    req = TradeOrderRequest(symbol=symbol, action=action, volume=volume, price=price, sl=sl, tp=tp, comment=comment)
    result = await execution.send_order(req)
    return result.dict()

@mcp.tool()
async def manage_trade(ticket: int, action: str, volume: float = None, sl: float = None, tp: float = None):
    """Gestiona posiciones: CLOSE, MODIFY_SL_TP, DELETE."""
    req = PositionManageRequest(ticket=ticket, action=action, volume=volume, sl=sl, tp=tp)
    result = await trade_mgmt.manage_position(req)
    return result.dict()

@mcp.tool()
async def get_account_status():
    """Estado financiero de la cuenta."""
    return await market.get_account_metrics()

@mcp.tool()
async def get_economic_calendar():
    """Eventos económicos de alto impacto próximos."""
    req = CalendarRequest(importance="HIGH", days_forward=1)
    return await calendar.get_economic_calendar(req)

# =============================================================================
# SYSTEM HEALTH
# =============================================================================

@mcp.resource("signals://live")
async def get_live_signals():
    """Stream de señales ZMQ disponibles."""
    return "Stream Active"

if __name__ == "__main__":
    import os
    import sys
    
    # Detectar si estamos en Docker/Linux (sin TTY) o en Windows (con TTY)
    # En Docker usamos HTTP para que el servidor persista
    # En Windows local usamos STDIO para integración directa con IDE
    is_docker = not sys.stdin.isatty() or os.environ.get("DOCKER_MODE", "").lower() == "true"
    
    if is_docker:
        logger.info("🐳 Modo Docker detectado - Iniciando servidor HTTP en puerto 8000...")
        mcp.run(transport="http", host="0.0.0.0", port=8000)
    else:
        logger.info("🖥️ Modo Windows detectado - Iniciando servidor STDIO...")
        mcp.run()
