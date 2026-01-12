@echo off
setlocal
cd /d "%~dp0"
title NEXUS COMMAND CENTER - STOP
echo Parando todo el sistema NEXUS...
python nexus_control.py stop
pause
