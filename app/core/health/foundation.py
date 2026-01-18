import time
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Health.Foundation")

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_HEALTH_ENDPOINT = "http://127.0.0.1:8000/health"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
LATENCY_WARNING_THRESHOLD_MS = 50

# =============================================================================
# MODELS
# =============================================================================
class HealthState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class SubsystemState(Enum):
    READY = "ready"
    WARMING_UP = "warming_up"
    NOT_READY = "not_ready"
    ERROR = "error"

@dataclass
class SubsystemCheck:
    name: str
    state: SubsystemState
    message: str
    latency_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class HealthReport:
    overall_state: HealthState
    timestamp: str
    subsystems: List[SubsystemCheck]
    endpoint_latency_ms: float
    warnings: List[str]
    is_ready_for_trading: bool

# =============================================================================
# TRANSPORT
# =============================================================================
async def request_with_retry(url: str, timeout: float = 5.0) -> Tuple[Optional[Dict], float, Optional[str]]:
    try: import httpx; has_httpx = True
    except: has_httpx = False
    
    if has_httpx:
        import httpx
        for i in range(MAX_RETRIES):
            try:
                s = time.perf_counter()
                async with httpx.AsyncClient() as c: r = await c.get(url, timeout=timeout)
                lat = (time.perf_counter()-s)*1000
                if r.status_code == 200: return r.json(), lat, None
                err = f"HTTP {r.status_code}"
            except Exception as e: err = str(e)
            await asyncio.sleep(2**i)
        return None, 0, f"Max retries: {err}"
    else:
        import urllib.request, json
        for i in range(MAX_RETRIES):
            try:
                s = time.perf_counter()
                with urllib.request.urlopen(url, timeout=timeout) as r: d = json.loads(r.read().decode())
                return d, (time.perf_counter()-s)*1000, None
            except Exception as e: err=str(e); time.sleep(2**i)
        return None, 0, f"Max retries: {err}"

# =============================================================================
# SCHEMA VALIDATION
# =============================================================================
def validate_health_schema(data: Dict) -> Tuple[bool, List[str]]:
    req = ["status"]
    miss = [f for f in req if f not in data]
    if miss: return False, miss
    warns = []
    if data.get("status") not in ["online", "ok"]: warns.append(f"Unexpected status: {data.get('status')}")
    if data.get("bridge_connected") is False: warns.append("MT5 bridge not connected")
    return len(miss) == 0, warns
