import zmq
import json
import time
from datetime import datetime
import os

# Colors for Terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def monitor():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Colors.HEADER}📡 FEAT NEXUS TELEMETRY MONITOR{Colors.ENDC}")
    print("Connecting to ZMQ BUS (5557)...")
    
    context = zmq.Context()
    
    # Subscribe to Python -> MT5 Channel
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5557")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "") # Listen to ALL
    
    print(f"{Colors.GREEN}✅ CONNECTED. Waiting for signals...{Colors.ENDC}")
    print("-" * 60)
    
    last_hud = 0
    
    while True:
        try:
            # Non-blocking check
            if subscriber.poll(100):
                msg = subscriber.recv_string()
                
                # Try JSON
                try:
                    # Sometimes message is "TOPIC JSON"
                    if " " in msg:
                        topic, json_str = msg.split(" ", 1)
                        data = json.loads(json_str)
                    else:
                        data = json.loads(msg)
                except:
                    data = {"raw": msg}
                
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Filter/Format
                msg_type = data.get("type", "UNKNOWN")
                
                if msg_type == "HEARTBEAT" or "equity" in data:
                    # Throttle HUD updates to 1 per second
                    if time.time() - last_hud > 1.0:
                        equity = data.get("equity", "N/A")
                        balance = data.get("balance", "N/A")
                        print(f"[{ts}] {Colors.BLUE}❤️ HUD: Eq ${equity} | Bal ${balance}{Colors.ENDC} | Conf: {data.get('confidence',0):.2f}")
                        last_hud = time.time()
                        
                elif msg_type in ["SIGNAL", "TRADE", "EXECUTION"]:
                    symbol = data.get("symbol", "N/A")
                    action = data.get("action", "SIGNAL")
                    color = Colors.GREEN if action == "BUY" else Colors.FAIL
                    print(f"[{ts}] {color}⚡ {action} {symbol} | Vol: {data.get('volume')} | {data.get('comment')}{Colors.ENDC}")
                    
                else:
                    print(f"[{ts}] 📩 {data}")
                    
        except KeyboardInterrupt:
            print("\n🛑 MONITOR STOPPED")
            break
        except Exception as e:
            print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
            time.sleep(1)

if __name__ == "__main__":
    monitor()
