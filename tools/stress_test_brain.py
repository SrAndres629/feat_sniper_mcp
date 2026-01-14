import asyncio
import logging
import sys
import os

# Set root dir
sys.path.append(os.getcwd())

from nexus_brain.inference_api import neural_api

async def run_stress_test():
    logging.basicConfig(level=logging.INFO)
    print("🚀 STARTING BRAIN STRESS TEST...")
    
    # Test 1: Zero Malformed Data
    zero_tick = {
        "bid": 0.0,
        "ask": 0.0,
        "tick_volume": 0.0,
        "symbol": "XAUUSD"
    }
    print("\n[TEST 1] Injecting Zero Tick...")
    res1 = await neural_api.predict_next_candle(zero_tick)
    print(f"Result: {res1}")

    # Test 2: NaN / None Values
    nan_tick = {
        "bid": None,
        "ask": 1.0, # Partial None
        "tick_volume": float('nan'),
        "symbol": "XAUUSD"
    }
    print("\n[TEST 2] Injecting NaN/None Tick...")
    res2 = await neural_api.predict_next_candle(nan_tick)
    print(f"Result: {res2}")

    # Test 3: Normal Data (to compare)
    normal_tick = {
        "bid": 2030.50,
        "ask": 2030.60,
        "tick_volume": 150,
        "symbol": "XAUUSD"
    }
    print("\n[TEST 3] Injecting Normal Tick...")
    res3 = await neural_api.predict_next_candle(normal_tick)
    print(f"Result: {res3}")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
