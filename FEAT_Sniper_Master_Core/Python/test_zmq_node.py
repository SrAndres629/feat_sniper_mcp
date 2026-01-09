import zmq
import time
import json
import random

def simulate_mt5_signal():
    """Simulates a signal being sent from MT5 via ZMQ."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    # Try to bind to port 5555 (MT5 would be the Publisher)
    # Note: If the MCP is the Subscriber, it should connect.
    # If the MCP is already binding, this will fail.
    # In zmq_bridge.py: self.socket = self.context.socket(zmq.SUB), self.socket.connect(self.addr)
    # This means the MCP is the SUBSCRIBER and it CONNECTS to the Publisher.
    # So MT5 (or this script) must BIND to the address.
    
    addr = "tcp://127.0.0.1:5555"
    print(f"Binding to {addr} as PUBLISHER...")
    try:
        socket.bind(addr)
    except Exception as e:
        print(f"Error binding to {addr}: {e}")
        print("Maybe another process is already binding to it?")
        return

    print("ZMQ Publisher ready. Sending signals every 2 seconds...")
    try:
        while True:
            data = {
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "action": "BUY",
                "price": 2045.50 + random.uniform(-1, 1),
                "confidence": 92.5,
                "reason": "Institutional Acceleration (Gas) confirmed at Demand Zone (Space)."
            }
            message = json.dumps(data)
            print(f"Sending: {message}")
            socket.send_string(message)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping simulation.")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    simulate_mt5_signal()
