@echo off
TITLE FEAT SNIPER - AI TOOLBOX (NO TRADING)
echo =================================================================
echo        FEAT SNIPER - MCP TOOLBOX (PASSIVE MODE)
echo        "Eyes Open, Weapons Holstered"
echo =================================================================
echo.
echo [INFO] This mode launches ONLY the AI tools and MT5 connection.
echo [INFO] The Trading Engine (Nexus) will NOT be started.
echo [INFO] Use this for reconnaissance, analysis, and debugging.
echo.

cd /d "%~dp0"
call .venv\Scripts\activate

echo [SYSTEM] Starting MCP Server in Standalone Mode...
python run_mcp_standalone.py

echo.
echo [SYSTEM] MCP Toolbox Shutdown.
pause
