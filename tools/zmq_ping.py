import zmq
import time
import sys

def main():
    context = zmq.Context()
    
    # PUB Socket (Sends Updates/PING to MT5)
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5556")
    print("[TEST] Bound PUB to 5556 (Sending PING)")
    
    # PULL Socket (Receives Commands/PONG from MT5)
    pull_socket = context.socket(zmq.PULL)
    pull_socket.bind("tcp://*:5555")
    print("[TEST] Bound PULL to 5555 (Listening for PONG)")
    
    print("[TEST] Waiting for MT5 connection... (Please Attach FEAT_Visualizer to Chart)")
    
    try:
        while True:
            # Send PING
            msg = '{"type":"PING", "timestamp":%f}' % time.time()
            pub_socket.send_string(msg)
            print(f"[TEST] Sent: {msg}")
            
            # Check for Reply (Non-blocking)
            try:
                reply = pull_socket.recv_string(flags=zmq.NOBLOCK)
                print(f"[TEST] RECEIVED: {reply}")
                if "PONG" in reply:
                    print("\n✅ SUCCESS: PING-PONG CONFIRMED!")
                    print("ZMQ Hyperbridge is FULLY OPERATIONAL.")
                    break
            except zmq.Again:
                pass
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("[TEST] Interrupted.")
    finally:
        pub_socket.close()
        pull_socket.close()
        context.term()

if __name__ == "__main__":
    main()
