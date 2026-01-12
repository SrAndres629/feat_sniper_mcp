from datetime import datetime
from typing import Generic, List, Literal, Optional, TypeVar, Any
from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")

# =============================================================================
# ENVELOPE PATTERN (RESPUESTA ESTNDAR)
# =============================================================================

class ErrorDetail(BaseModel):
    code: str
    message: str
    suggestion: str

class ResponseModel(BaseModel, Generic[T]):
    """
    Modelo de respuesta global. Garantiza que n8n siempre reciba
    la misma estructura, facilitando el parseo y la resiliencia.
    """
    status: Literal["success", "error"]
    data: Optional[T] = None
    error: Optional[ErrorDetail] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# =============================================================================
# INPUT MODELS (VALIDACIN DE ENTRADA)
# =============================================================================

class MarketDataRequest(BaseModel):
    symbol: str
    timeframe: str = "M5"
    n_candles: int = Field(default=100, ge=1, le=1000)
    output_format: Literal["json", "csv"] = "json"

class TradeOrderRequest(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"]
    volume: float = Field(..., ge=0.01)
    price: Optional[float] = None  # Requerido para rdenes pendientes
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: str = "AI_MCP_Order"

class IndicatorRequest(BaseModel):
    symbol: str
    timeframe: str = "M15"
    indicator: Literal["RSI", "MACD", "MA", "ATR", "BOLLINGER"]
    period: int = Field(default=14, ge=1)
    ma_method: Optional[int] = 0  # 0: SMA, 1: EMA, etc.
    ma_price: Optional[int] = 0    # 0: Close, 1: Open, etc.

class MQL5CodeRequest(BaseModel):
    name: str  # Nombre del archivo (ej: 'SuperTrend_AI')
    code: str  # Cdigo MQL5 completo
    compile: bool = True

class CalendarRequest(BaseModel):
    currency: Optional[str] = None
    importance: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    days_forward: int = Field(default=1, ge=0, le=7)

class HistoryRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)

class PanoramaRequest(BaseModel):
    resize_factor: float = Field(default=0.75, ge=0.1, le=1.0)

class VolatilityRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"
    period: int = Field(default=14, ge=1, le=100)

class PositionManageRequest(BaseModel):
    ticket: int
    action: Literal["CLOSE", "MODIFY", "DELETE"]  # DELETE para rdenes pendientes
    volume: Optional[float] = None  # Requerido para CLOSE parcial
    sl: Optional[float] = None      # Requerido para MODIFY
    tp: Optional[float] = None      # Requerido para MODIFY
    price: Optional[float] = None   # Para modificar precio de rdenes pendientes

# =============================================================================
# OUTPUT MODELS (DATOS LIMPIOS PARA n8n)
# =============================================================================

class AccountMetrics(BaseModel):
    balance: float
    equity: float
    margin_free: float
    margin_level: float
    profit: float
    positions_total: int
    currency: str
    server: str

class Candle(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    tick_volume: int

class MarketDataResponse(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]

class TradeOrderResponse(BaseModel):
    ticket: int
    symbol: str
    action: str
    price: float
    volume: float
    
class VisionResponse(BaseModel):
    width: int
    height: int
    format: str
    image_base64: str

class VolatilityMetrics(BaseModel):
    symbol: str
    atr: float
    spread: float
    spread_points: int
    volatility_status: str  # HIGH, NORMAL, LOW

class PositionActionResponse(BaseModel):
    ticket: int
    status: str
    message: str
