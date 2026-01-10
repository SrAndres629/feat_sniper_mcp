@echo off
setlocal
cd /d "%~dp0"
title NEXUS COMMAND CENTER - START
cls
echo ============================================================
echo      FEAT SNIPER NEXUS - INICIO INSTITUCIONAL
echo ============================================================
echo.
python nexus_control.py start
pause
