@echo off
setlocal
cd /d "%~dp0"
title FEAT SNIPER NEXUS - MASTER COMMAND
color 0B

:: Ensure Python is in path
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b
)

if "%1"=="stop" (
    python nexus_control.py stop
    goto :EOF
)

if "%1"=="audit" (
    python nexus_control.py audit
    goto :EOF
)

:: Default: Start
python nexus_control.py start
pause
