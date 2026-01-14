import zmq
import time
import json
import random
import math
from datetime import datetime

# Config
ZMQ_PORT = 5555
TICKS_PER_SEC = 50 # Moderate HFT

def run_simulation():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    
    print(f"⚔️ COMBAT SIMULATOR v1.0 INITIALIZED (Target: {ZMQ_PORT})")
    print(f"🌊 Injecting High-Frequency Data ({TICKS_PER_SEC} ticks/sec)...")
    print("Simulating XAUUSD Market...\n")
    
    price = 2000.0
    tick_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Physics Mock: Sine Wave + Noise + Random Spikes
            t = time.time() - start_time
            noise = random.uniform(-0.2, 0.2)
            trend_val = math.sin(t / 10.0) * 1.0 # Slow wave over 60s
            
            # Spike Injection every 15 seconds
            # Cycle 15s. If t % 15 < 0.2 (short burst)
            is_spike = (t % 15.0) < 0.2 
            
            if is_spike:
                vol = 5000.0 # Huge Volume (Accel Trigger)
                # Strong Move
                price_jump = 2.0 if math.sin(t) > 0 else -2.0
            else:
                vol = random.uniform(50, 150)
                price_jump = random.uniform(-0.05, 0.05)
            
            price += (trend_val * 0.001) + price_jump + noise
            
            tick = {
                "symbol": "XAUUSD",
                "bid": round(price, 2),
                "ask": round(price + 0.20, 2),
                "tick_volume": vol,
                "real_volume": vol,
                "timestamp": datetime.now().timestamp(),
                "simulated_time": datetime.now().isoformat()
            }
            
            socket.send_string(json.dumps(tick))
            
            tick_count += 1
            if tick_count % 50 == 0:
                print(f"\r[SIM] Ticks: {tick_count} | Price: {price:.2f} | Vol: {vol:.0f} {'⚡' if is_spike else ''}   ", end="")
            
            time.sleep(1.0 / TICKS_PER_SEC)
            
    except KeyboardInterrupt:
        print("\n\n🏳️ Simulation Stopped.")
        socket.close()
        context.term()

if __name__ == "__main__":
    run_simulation()
