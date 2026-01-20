"""
FEAT SNIPER: MT5 TICK LISTENER
==============================
High-frequency worker that pulls real-time ticks from MT5 and updates the buffers.
Implements the 'Zero-Lag' requirement for microstructure sensors.
"""

import time
import threading
import logging
import numpy as np
from typing import Optional
from app.core.mt5_conn.utils import mt5, MT5_AVAILABLE
from app.core.config import settings
from nexus_core.microstructure.ticker import tick_buffer
from nexus_core.microstructure.scanner import micro_scanner

logger = logging.getLogger("FEAT.TickListener")

class TickListener(threading.Thread):
    def __init__(self, symbol: str = "XAUUSD"):
        super().__init__(daemon=True, name=f"TickListener-{symbol}")
        self.symbol = symbol
        self.running = False
        self.last_tick_time = 0
        
    def stop(self):
        self.running = False
        
    def run(self):
        if not MT5_AVAILABLE:
            logger.error("❌ MT5 not available. TickListener cannot start.")
            return

        logger.info(f"📡 TICK LISTENER: Monitoring {self.symbol} in High-Frequency mode.")
        self.running = True
        
        # Ensure symbol is selected
        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"❌ Failed to select symbol {self.symbol}")
            return
            
        while self.running:
            try:
                # 1. Fetch latest ticks since last check
                # MT5 returns ticks in a specific struct
                ticks = mt5.copy_ticks_from(self.symbol, int(time.time() * 1000), 10, mt5.COPY_TICKS_ALL)
                
                if ticks is not None and len(ticks) > 0:
                    for t in ticks:
                        if t[0] > self.last_tick_time: # index 0 is time_msc
                            # Map MT5 tick to our internal format
                            tick_data = {
                                'time': t[0],
                                'bid': t[1], # bid
                                'ask': t[2], # ask
                                'bid_vol': t[5], # volume (bid side proxy)
                                'ask_vol': t[5]  # MT5 tick_volume is usually total
                            }
                            tick_buffer.add_tick(tick_data)
                            self.last_tick_time = t[0]
                    
                    # 2. Trigger instant Microstructure Scan if buffer is healthy
                    if tick_buffer.ready:
                        start_time = time.perf_counter()
                        micro_scanner.live_scan()
                        end_time = time.perf_counter()
                        
                        latency_ms = (end_time - start_time) * 1000
                        if latency_ms > 5:
                            logger.warning(f"⚠️ Micro-Scanner Latency high: {latency_ms:.2f}ms")
                
                # Sleep briefly to avoid 100% CPU, but keep it HFT-friendly
                time.sleep(0.001) # 1ms poll rate
                
            except Exception as e:
                logger.error(f"💥 TickListener Error: {e}")
                time.sleep(1)

# Singleton instance
tick_listener = TickListener(symbol=settings.TRADE_SYMBOL if hasattr(settings, 'TRADE_SYMBOL') else "XAUUSD")
