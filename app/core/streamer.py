import logging
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from app.core.config import settings

# Logger for internal streamer errors (doesn't stream to itself)
logger = logging.getLogger("feat.streamer")

class SupabaseStreamer(logging.Handler):
    """
    Broadcaster that pushes Logs and State to Supabase Realtime.
    Acts as a logging.Handler to intercept log records.
    """
    def __init__(self, supabase_client=None):
        super().__init__()
        self.client = supabase_client
        self.buffer = []
        self.last_flush = time.time()
        self.flush_interval = 2.0 # Seconds
        self.enabled = settings.SUPABASE_URL and settings.SUPABASE_KEY
        
        # Async loop reference
        self.loop = None
        
        if not self.enabled:
            logger.warning("⚠️ Supabase Credentials missing. Dashboard Streamer DISABLED.")
        
        # Throttling and Concurrency Control (Windows Stabilization)
        self._locks = {} # table_name -> is_busy
        self._last_push = {} # table_name -> timestamp
        self.throttle_rate = 1.0 # 1 second minimum between pushes for metrics/signals

    def emit(self, record):
        """Intercepts a log record."""
        if not self.enabled: return
        
        # Filter: Only INFO or higher for "bot_activity_log" (Clean logs)
        if record.levelno < logging.INFO: return

        try:
            msg = self.format(record)
            
            # Structuring the log payload
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": record.name,
                "message": f"[{record.levelname}] {msg}", 
                "session_id": settings.MT5_LOGIN or "UNKNOWN",
                "level": record.levelname
            }
            
            self.buffer.append(log_entry)
            
            # For errors, flush immediately but safely via threadsafe call if loop is ready
            if record.levelno >= logging.ERROR and self.loop:
                asyncio.run_coroutine_threadsafe(self.flush_async(), self.loop)
                
        except Exception:
            self.handleError(record)

    def flush_sync(self):
        """Synchronous flush to bot_activity_log."""
        if not self.buffer or not self.client: return
        try:
            data = self.buffer.copy()
            self.buffer.clear()
            self.client.table("bot_activity_log").insert(data).execute()
        except Exception as e:
            pass # Silent fail in sync mode to prevent loops

    async def start_async_loop(self):
        self.loop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush_async()

    async def _safe_execute(self, table: str, operation, throttle: bool = False):
        """Executes operation with throttling, serialization, and error suppression."""
        if not self.client: return
        
        now = time.time()
        
        # 1. Throttling (for high-frequency telemetry)
        if throttle:
            if table in self._last_push and (now - self._last_push[table]) < self.throttle_rate:
                return
            self._last_push[table] = now

        # 2. Concurrency Lock (Serial execution per table)
        if self._locks.get(table):
            return 
        
        self._locks[table] = True
        try:
             await self.loop.run_in_executor(None, operation)
        except Exception as e:
            err_msg = str(e)
            # Socket Error 10035 (WSAEWOULDBLOCK) or Supabase missing columns/tables
            if "10035" in err_msg or "PGRST205" in err_msg or "PGRST204" in err_msg:
                 pass 
            else:
                 logger.error(f"!! STREAMER ERROR ({table}): {err_msg}")
        finally:
            self._locks[table] = False

    async def flush_async(self):
        """Async flush to bot_activity_log."""
        if not self.buffer or not self.client: return
        data = self.buffer.copy()
        self.buffer.clear()
        await self._safe_execute("bot_activity_log", lambda: self.client.table("bot_activity_log").insert(data).execute())

    async def push_metrics(self, data: Dict[str, Any]):
        """Pushes to 'live_metrics' with throttling."""
        if not self.enabled: return
        payload = {"updated_at": datetime.now().isoformat(), "session_id": settings.MT5_LOGIN, **data}
        await self._safe_execute("live_metrics", 
                               lambda: self.client.table("live_metrics").upsert(payload, on_conflict="session_id").execute(),
                               throttle=True)

    async def push_signals(self, data: Dict[str, Any]):
        """Pushes to 'neural_signals' with throttling."""
        if not self.enabled: return
        payload = {"created_at": datetime.now().isoformat(), "session_id": settings.MT5_LOGIN, **data}
        await self._safe_execute("neural_signals", 
                               lambda: self.client.table("neural_signals").insert(payload).execute(),
                               throttle=True)

# Global instance
streamer = None

def init_streamer():
    """Initializes the global streamer instance."""
    global streamer
    try:
        from supabase import create_client, Client
        
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            streamer = SupabaseStreamer(client)
            return streamer
    except ImportError:
        logger.error("❌ 'supabase' library not installed. Run: pip install supabase")
    except Exception as e:
        logger.error(f"❌ Failed to init Supabase: {e}")
    
    return None
