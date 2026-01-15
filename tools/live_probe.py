
import asyncio
import zmq
import zmq.asyncio
import json
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ZMQ_Probe")

async def probe_zmq():
    try:
        context = zmq.asyncio.Context()
        subscriber = context.socket(zmq.SUB)
        
        # Port 5555 is usually the DATA/TICK stream from MT5
        subscriber.connect("tcp://127.0.0.1:5555")
        subscriber.subscribe("")
        
        logger.info("📡 Probing ZMQ Port 5555 (Waiting 5s for ticks)...")
        
        try:
            # Wait for a message with a 5 second timeout
            msg = await asyncio.wait_for(subscriber.recv_string(), timeout=5.0)
            logger.info(f"✅ DATA RECEIVED: {msg[:100]}...") # Print first 100 chars
            return True
        except asyncio.TimeoutError:
            logger.error("❌ TIMEOUT: No data received on port 5555 (MT5 Bridge Silent/Down).")
            return False
        finally:
            subscriber.close()
            context.term()
            
    except Exception as e:
        logger.error(f"❌ PROBE ERROR: {e}")
        return False

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(probe_zmq())
    if result:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
