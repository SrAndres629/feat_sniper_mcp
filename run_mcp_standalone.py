import os
import sys
import logging

# Configure Logging for the Launcher
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [LAUNCHER] | %(message)s')
logger = logging.getLogger("Launcher")

def main():
    logger.info("🔧 FEAT SNIPER: Initializing Standalone MCP Toolbox...")
    
    # Set the Standalone Flag
    os.environ["FEAT_MCP_STANDALONE"] = "1"
    
    # Import and Run current MCP Server
    # We import here so the env var is set before module load (if needed)
    try:
        from mcp_server import mcp
        logger.info("🚀 Launching MCP Server in Passive Mode...")
        mcp.run()
    except ImportError as e:
        logger.error(f"❌ Failed to import MCP Server: {e}")
    except Exception as e:
        logger.error(f"❌ Runtime Error: {e}")

if __name__ == "__main__":
    main()
