@echo off
TITLE FEAT NEXUS - IMMORTAL CORE DAEMON
SETLOCAL EnableDelayedExpansion

:: Premium UI
COLOR 0A
cls
echo =================================================================
echo        FEAT SNIPER v2.1 - NEURAL EVOLUTION SYSTEM
echo        "Si Vis Pacem, Para Bellum"
echo =================================================================
echo.

:: 1. Virtual Environment Activation
echo [SYSTEM] Locating Virtual Environment...
if NOT exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual Environment not found in .venv/
    echo [FIX] Please initialize: python -m venv .venv
    pause
    exit /b
)
call .venv\Scripts\activate
echo [SYSTEM] Environment: ACTIVATED
echo.

:: 2. Pre-Flight Checks
echo [SYSTEM] Starting Atomic Integrity Scans...

:: Check PyTorch & CUDA
python -c "import torch; print(f' - Neural Backend: PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
if errorlevel 1 (
    echo [FATAL] PyTorch verify failed. Check installation.
    pause
    exit /b
)

:: Check MT5 Library
python -c "import MetaTrader5 as mt5; print(f' - Broker Interface: MT5 Library {mt5.__version__}')"
if errorlevel 1 (
    echo [FATAL] MT5 library verify failed.
    pause
    exit /b
)

:: 3. Market Awareness
python -c "from datetime import datetime; today = datetime.now().weekday(); print(' - Market Status: [WEEKEND]' if today >= 5 else ' - Market Status: [OPEN]')"

:: 4. Launch Sequence
echo.
echo [SYSTEM] All systems nominal. Launching Immortal Daemon...
echo -----------------------------------------------------------------
python nexus_daemon.py

:: 5. Exit Handling
echo.
echo -----------------------------------------------------------------
echo [SYSTEM] Daemon suspended.
pause
