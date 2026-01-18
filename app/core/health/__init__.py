import asyncio
from .sentinel import health_sentinel, HealthSentinel
from .foundation import HealthState, SubsystemState, HealthReport, SubsystemCheck, DEFAULT_HEALTH_ENDPOINT

async def run_boot_diagnostic(url: str = DEFAULT_HEALTH_ENDPOINT, print_report: bool = True) -> bool:
    r = await health_sentinel.run_full_diagnostic(url)
    if print_report: health_sentinel.print_health_card(r)
    return r.is_ready_for_trading

def run_diagnostic_sync(url: str = DEFAULT_HEALTH_ENDPOINT) -> bool:
    return asyncio.run(run_boot_diagnostic(url))

__all__ = ["health_sentinel", "run_boot_diagnostic", "run_diagnostic_sync", "HealthSentinel", "HealthState", "SubsystemState", "HealthReport", "SubsystemCheck"]
