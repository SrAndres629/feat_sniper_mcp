@echo off
setlocal enabledelayedexpansion
title FEAT Sniper MCP - System Level
color 0B

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating .venv...
    call .venv\Scripts\activate.bat
)

echo [INFO] Starting MCP Server (Stdio Transport)...
echo [INFO] Logging redirected to stderr to preserve Protocol.

python mcp_server.py

pause
