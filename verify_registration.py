import sys
import os
from pathlib import Path

# Add root to sys.path
sys.path.append(os.getcwd())

from mcp_server import mcp

async def test_registration():
    print("=== Herramientas Registradas en MCP ===")
    tools = await mcp.get_tools()
    # In FastMCP, tools might be a list of Tool objects or strings depending on version.
    # Let's try to handle both or just use names if they are already strings.
    names = []
    for t in tools:
        if isinstance(t, str):
            names.append(t)
        elif hasattr(t, "name"):
            names.append(t.name)
        elif isinstance(t, dict) and "name" in t:
            names.append(t["name"])
            
    registered_skills = [n for n in names if n.startswith("skill_")]
    
    for name in names:
        print(f"- {name}")
        
    if registered_skills:
        print(f"\n✅ Se encontraron {len(registered_skills)} skills personalizadas.")
    else:
        print("\n❌ No se encontraron skills personalizadas. Verifique el directorio 'FEAT_Sniper_Master_Core' y los archivos .mq5.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_registration())
