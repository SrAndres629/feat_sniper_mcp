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

    def emit(self, record):
        """Intercepts a log record."""
        if not self.enabled: return
        
        # Filter: Only INFO or higher for "audit_logs" (Clean logs)
        if record.levelno < logging.INFO: return

        try:
            msg = self.format(record)
            
            # Structuring the log payload
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "module": record.name,
                "message": msg,
                "session_id": settings.MT5_LOGIN or "UNKNOWN"
            }
            
            self.buffer.append(log_entry)
            
            if record.levelno >= logging.ERROR:
                self.flush_sync()
                
        except Exception:
            self.handleError(record)

    def flush_sync(self):
        """Synchronous flush to audit_logs."""
        if not self.buffer or not self.client: return
        try:
            data = self.buffer.copy()
            self.buffer.clear()
            self.client.table("audit_logs").insert(data).execute()
        except Exception as e:
            print(f"!! STREAMER ERROR: {e}")

    async def start_async_loop(self):
        self.loop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush_async()

    async def _safe_execute(self, table: str, operation):
        """Executes operation with error suppression for missing tables."""
        try:
             await self.loop.run_in_executor(None, operation)
        except Exception as e:
            if "PGRST205" in str(e): # PostgREST code for table not found
                 # Suppress repeated errors for missing tables to avoid spam
                 pass 
            else:
                 print(f"!! STREAMER ERROR ({table}): {e}")

    async def flush_async(self):
        """Async flush to audit_logs."""
        if not self.buffer or not self.client: return
        data = self.buffer.copy()
        self.buffer.clear()
        await self._safe_execute("audit_logs", lambda: self.client.table("audit_logs").insert(data).execute())

    async def push_metrics(self, data: Dict[str, Any]):
        """Pushes to 'live_metrics'."""
        if not self.enabled or not self.client: return
        payload = {"updated_at": datetime.now().isoformat(), "session_id": settings.MT5_LOGIN, **data}
        await self._safe_execute("live_metrics", lambda: self.client.table("live_metrics").upsert(payload, on_conflict="session_id").execute())

    async def push_signals(self, data: Dict[str, Any]):
        """Pushes to 'neural_signals'."""
        if not self.enabled or not self.client: return
        payload = {"created_at": datetime.now().isoformat(), "session_id": settings.MT5_LOGIN, **data}
        await self._safe_execute("neural_signals", lambda: self.client.table("neural_signals").insert(payload).execute())

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
