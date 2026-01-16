import asyncio
import logging
import time
from typing import Optional, Any
from datetime import datetime

import json
import os

class PersistedTokenBucket:
    """
    [SENIOR ARCHITECTURE] Token Bucket with Persistent State.
    Ensures that rate limits and punishments survive process crashes/restarts.
    """
    def __init__(self, filename="data/rate_limit.json", capacity=10, fill_rate=0.5):
        self.filename = filename
        self.capacity = capacity
        self.fill_rate = fill_rate # tokens per second (0.5 = 1 every 2s)
        self.tokens = capacity
        self.last_update = time.time()
        self._load_state()

    def _load_state(self):
        """Loads bucket state from disk."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    state = json.load(f)
                    self.tokens = state.get('tokens', self.capacity)
                    self.last_update = state.get('last_update', time.time())
                    # Ensure tokens are up to date with time elapsed since last save
                    self.consume(0) 
            except Exception as e:
                logger.error(f"Failed to load RateLimit state: {e}")

    def _save_state(self):
        """Saves current state to disk (Async or fast sync)."""
        try:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, 'w') as f:
                json.dump({
                    'tokens': self.tokens,
                    'last_update': self.last_update
                }, f)
        except Exception as e:
            logger.error(f"Failed to save RateLimit state: {e}")

    def consume(self, amount=1) -> bool:
        """Attempts to consume tokens. Returns True if successful."""
        now = time.time()
        time_passed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + time_passed * self.fill_rate)
        self.last_update = now

        if self.tokens >= amount:
            self.tokens -= amount
            self._save_state()
            return True
        return False

    def is_locked(self) -> bool:
        """Returns True if the bucket is empty (rate limited)."""
        return self.tokens < 1

class CircuitBreaker:
    """
    Módulo 9: The Circuit Breaker.
    Monitores de seguridad SRE.
    """
    def __init__(self):
        self.last_tick_time = time.time()
        self.is_tripped = False
        self.current_level = 0  # 0: Safe, 1: Defensive, 2: Survival, 3: Shutdown
        self.pause_until: float = 0.0 # Multi-level operational pause
        
        # [SENIOR FIX] Persisted Rate Limiter
        self.rate_limiter = PersistedTokenBucket(capacity=20, fill_rate=0.5) # 1 order / 2s
        
        # Tolerance: 15 seconds of silence = DEATH
        self.max_latency = 15.0 
        self.trade_manager: Optional[Any] = None 
        self._daily_opening_balance: Optional[float] = None

    def set_trade_manager(self, manager):
        self.trade_manager = manager

    async def get_drawdown(self) -> float:
        """Calcula el drawdown diario actual."""
        from app.core.config import settings
        from app.core.mt5_conn import mt5_conn
        import MetaTrader5 as mt5
        
        # Sincronizar balance inicial si es necesario
        if not self._daily_opening_balance:
            account = await mt5_conn.execute(mt5.account_info)
            if account:
                self._daily_opening_balance = account.balance
        
        if not self._daily_opening_balance:
            return 0.0
            
        account = await mt5_conn.execute(mt5.account_info)
        if not account:
            return 0.0
            
        real_loss = self._daily_opening_balance - account.equity
        return (real_loss / self._daily_opening_balance) * 100 if self._daily_opening_balance > 0 else 0.0

    async def get_lot_multiplier(self) -> float:
        """
        Retorna el multiplicador de lotaje basado en el nivel de riesgo.
        Refinado por Gemini Web (Visionary): 2%/4%/6%
        """
        from app.core.config import settings
        
        # Check Operational Pause
        if time.time() < self.pause_until:
            logger.warning(f"⏳ OPERATIONAL PAUSE ACTIVE: Resuming in {int(self.pause_until - time.time())}s")
            return 0.0

        dd = await self.get_drawdown()
        
        if dd >= settings.CB_LEVEL_3_DD:
            self.current_level = 3
            if not self.is_tripped:
                await self.emergency_shutdown()
            return 0.0
        elif dd >= settings.CB_LEVEL_2_DD:
            self.current_level = 2
            if self.pause_until == 0:
                self.pause_until = time.time() + 3600 # 1h Operational Pause
                logger.critical("🚨 LEVEL 2 DD REACHED: Triggering 1-hour operational pause.")
            return 0.50  # 50% Lot Size (Visionary Target)
        elif dd >= settings.CB_LEVEL_1_DD:
            self.current_level = 1
            return 0.75  # 75% Lot Size (Visionary Target)
        
        self.current_level = 0
        return 1.0

    def can_execute(self) -> bool:
        """
        Veredicto final para la ejecución de órdenes.
        Refined with [SENIOR] Persisted Rate Limiting.
        """
        if self.is_tripped or self.current_level >= 3:
            return False
            
        if time.time() < self.pause_until:
             return False
             
        # Check Physical Rate Limiter
        if self.rate_limiter.is_locked():
            logger.error("🛑 PHYSICAL RATE LIMITER ACTIVE: Blocking Order Frequency.")
            return False
            
        # Consume token on check (optimistic lock)
        return self.rate_limiter.consume(1)

    def get_hierarchical_status(self) -> Dict[str, Any]:
        """Returns full status for HUD/Orchestrator."""
        return {
            "is_tripped": self.is_tripped,
            "level": self.current_level,
            "can_trade": self.can_execute(),
            "pause_remaining": max(0, int(self.pause_until - time.time())),
            "tokens_remaining": round(self.rate_limiter.tokens, 2),
            "timestamp": datetime.now().isoformat()
        }

    def record_failure(self):
        """Registra un fallo técnico (broker error, timeout)."""
        if not hasattr(self, '_consecutive_failures'):
            self._consecutive_failures = 0
        self._consecutive_failures += 1
        logger.warning(f"[CB] Technical failure recorded ({self._consecutive_failures} consecutive)")
        
        # Trip circuit after 5 consecutive failures
        if self._consecutive_failures >= 5:
            logger.critical(f"[CB] Too many failures ({self._consecutive_failures}) - triggering protective shutdown")
            asyncio.create_task(self.emergency_shutdown())

    def record_success(self):
        """Limpia contadores de fallos temporales."""
        if not hasattr(self, '_consecutive_failures'):
            self._consecutive_failures = 0
        if self._consecutive_failures > 0:
            logger.info(f"[CB] Success recorded - resetting failure counter from {self._consecutive_failures}")
            self._consecutive_failures = 0

    def heartbeat(self):
        """Llamado por cada tick ZMQ recibido."""
        self.last_tick_time = time.time()

    async def monitor_heartbeat(self):
        """Tarea de fondo: Monitorea latencia y resets diarios."""
        logger.info(f"🛡️ Multi-Level Circuit Breaker ARMED (Visionary Strategy Active)")
        try:
            while True:
                # 1. Monitoreo de Latencia (Heartbeat)
                latency = time.time() - self.last_tick_time
                if latency > self.max_latency and not self.is_tripped:
                    logger.critical(f"🚨 CIRCUIT TRIP: Dead Silence ({latency:.1f}s > {self.max_latency}s)")
                    await self.emergency_shutdown()
                
                # 2. Monitoreo de Drawdown (Status Check)
                multiplier = await self.get_lot_multiplier()
                if self.current_level > 0:
                     status_msg = f"⚠️ CB STATUS: Level {self.current_level}"
                     if self.pause_until > time.time():
                         status_msg += f" (PAUSED for {int(self.pause_until - time.time())}s)"
                     logger.warning(status_msg)

                await asyncio.sleep(5.0)  # Check every 5s
        except asyncio.CancelledError:
            logger.info("🛡️ Circuit Breaker Disarmed")

    async def emergency_shutdown(self):
        self.is_tripped = True
        self.current_level = 3
        logger.critical("⛔ SRE PROTOCOL: EXECUTE ORDER 66 (TOTAL HALT)")
        if self.trade_manager:
            try:
                # Cerramos todo
                await self.trade_manager.close_all_positions()
            except Exception as e:
                logger.error(f"Emergency Execution Failed: {e}")
        else:
            logger.error("SRE Error: TradeManager not linked! Manual intervention required.")

# Singleton Global
circuit_breaker = CircuitBreaker()
