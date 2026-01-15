
import sys
import os

# Adjust path to include root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("--- VERIFICATION START ---")
print(f"Platform: {sys.platform}")
print(f"Stdout Encoding: {sys.stdout.encoding}")

try:
    print("Importing mcp_server...")
    import mcp_server
    print("mcp_server imported successfully.")
except Exception as e:
    print(f"FAIL: Could not import mcp_server: {e}")
    sys.exit(1)

if mcp_server.hybrid_model:
    print("SUCCESS: HybridModel initialized successfully.")
    print(f"Model: {mcp_server.hybrid_model}")
else:
    print("WARNING: HybridModel is None (might be expected if model file missing, but checked code handles it safe now).")

try:
    print("Checking registered tools...")
    # FastMCP typically stores tools in a registry.
    # Check mcp object attributes.
    mcp = mcp_server.mcp
    # Depending on FastMCP version, it might be in _tools or tools
    tools_count = len(getattr(mcp, '_tools', getattr(mcp, 'tools', [])))
    print(f"Registered Tools Count: {tools_count}")
    if tools_count == 0:
        print(f"MCP Object Dir: {dir(mcp)}")
        # Check if _function_registry exists (common in some MCP libs)
        # or list_tools()
        if hasattr(mcp, 'list_tools'):
             try:
                 print(f"List Tools (Async check needed maybe?): {mcp.list_tools()}")
             except:
                 print("list_tools() failed or needs async.")
        
    if tools_count > 5:
        print("SUCCESS: Master Tools detected.")
        print(f"Tools: {[t.name for t in getattr(mcp, '_tools', getattr(mcp, 'tools', []))]}")
    else:
        print("WARNING: Low tool count. Check registration.")
except Exception as e:
    print(f"FAIL: Could not verify tools: {e}")


print("--- VERIFICATION END ---")
