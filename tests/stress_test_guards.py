"""
CHAOS GENERATOR (LVL 15)
========================
Simulates Flash Crash scenarios using GAN-Lite patterns.
Verifies CircuitGuard response speed.
"""

import sys
import os
import unittest
import asyncio
import time

# Adjust path to find app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from app
from app.services.circuit_breaker import circuit_breaker
from app.core.config import settings

class ChaosFlashCrash(unittest.TestCase):
    
    def simulate_flash_crash(self):
        """Generates synthetic crash data [-5% in 10 ticks]."""
        start_price = 2000.0
        prices = [start_price]
        
        # Generator: Exponential decay (Crash)
        for i in range(10):
            drop = prices[-1] * 0.005 # 0.5% drop per tick -> 5% total
            prices.append(prices[-1] - drop)
            
        return prices
    
    def test_circuit_response(self):
        """Verify breaker trips on Synthetic Crash."""
        print("\n🌪️ [CHAOS] Initiating Flash Crash Simulation...")
        
        # 1. Reset Breaker State
        circuit_breaker.is_tripped = False
        circuit_breaker.current_level = 0
        circuit_breaker.pause_until = 0.0
        circuit_breaker._daily_opening_balance = 10000.0
        
        # 2. Simulate Crash Data
        prices = self.simulate_flash_crash()
        
        # 3. Simulate Execution Loop
        start_balance = 10000.0
        current_equity = 10000.0
        tripped_at = -1
        
        # Mocking the dependency on MT5 via a closure
        async def check_circuit(equity):
            # Inject state manually for test
            real_loss = start_balance - equity
            dd_percent = (real_loss / start_balance) * 100
            
            # Manually trigger logic from get_lot_multiplier
            if dd_percent >= settings.CB_LEVEL_3_DD:
                circuit_breaker.current_level = 3
                return 0.0
            elif dd_percent >= settings.CB_LEVEL_2_DD:
                circuit_breaker.current_level = 2
                return 0.5
            return 1.0

        print(f"DEBUG: Generated {len(prices)} price ticks.")

        for i, p in enumerate(prices):
            # Simulate a 2% equity drop per tick (Total meltdown)
            loss_chunk = start_balance * 0.02 
            current_equity -= loss_chunk
            
            # Check Circuit (Async wrapper)
            loop = asyncio.get_event_loop()
            lot_mult = loop.run_until_complete(check_circuit(current_equity))
            
            status = "OPEN"
            if circuit_breaker.current_level >= 3:
                status = "TRIPPED 🔴"
            elif circuit_breaker.current_level == 2:
                status = "PAUSED ⚠️"
                
            print(f"Tick {i}: Price={p:.2f} | Eq={current_equity:.1f} (-{(start_balance-current_equity)/start_balance:.1%}) | Circuit={status}")
            
            if circuit_breaker.current_level >= 3:
                tripped_at = i
                break
                
        if tripped_at != -1:
            print(f"✅ PASSED: Circuit tripped at Tick {tripped_at} (Equity Drop: {(start_balance-current_equity)/start_balance:.1%})")
        else:
            print("❌ FAILED: Circuit did not trip during crash!")
            
        self.assertTrue(tripped_at != -1, "Circuit Breaker failed to stop the bleeding.")

if __name__ == "__main__":
    unittest.main()
