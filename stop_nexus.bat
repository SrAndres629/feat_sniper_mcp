@echo off
setlocal
title FEAT NEXUS - TERMINATOR

echo [STOP] Initiating Global Protocol 66...

REM 1. Kill Streamlit specifically
taskkill /F /IM python.exe /FI "WINDOWTITLE eq VISUAL CORTEX*" >nul 2>&1

REM 2. Kill FEAT NEXUS core
taskkill /F /IM python.exe /FI "WINDOWTITLE eq FEAT NEXUS*" >nul 2>&1

REM 3. General Python cleanup (Risk: might kill other bots, but user requested 'everything')
REM taskkill /F /IM python.exe /T >nul 2>&1

echo [STATUS] Components Terminated.
pause
