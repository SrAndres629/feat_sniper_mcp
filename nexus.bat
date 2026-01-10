@echo off
setlocal
cd /d "%~dp0"
title NEXUS COMMAND CENTER - MASTER ORCHESTRATOR
color 0A
cls

echo ==============================================================================
echo.
echo      NN   NN  EEEEEEE  XX   XX  UU   UU   SSSSS   PRIMME
echo      NNN  NN  EE        XX XX   UU   UU  SS       P    P
echo      NN N NN  EEEEE      XXX    UU   UU   SSSSS   PPPPPP
echo      NN  NNN  EE        XX XX   UU   UU       SS  P
echo      NN   NN  EEEEEEE  XX   XX   UUUUU    SSSSS   P
echo.
echo      [ MASTER AUDIT & SELF-HEALING PROTOCOL v2.5 ]
echo ==============================================================================
echo.
echo [INFO] Inicializando entorno institucional...
echo [INFO] Cargando configuraciones de seguridad...

choice /C SAQ /N /M "[S] Start System  [A] Audit Only  [Q] Quit : "

if errorlevel 3 goto QUIT
if errorlevel 2 goto AUDIT
if errorlevel 1 goto START

:START
cls
echo [EXEC] Iniciando Secuencia Maestra de Arranque...
python nexus_control.py start
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo [FATAL] El sistema no pudo arrancar. Revisa los logs de "Deep Healer".
    pause
    color 0A
    goto END
)
echo.
echo [SUCCESS] Sistema estabilizado.
goto END

:AUDIT
cls
echo [EXEC] Ejecutando Auditoria Profunda (Sin Arranque)...
python nexus_control.py audit
echo.
pause
goto END

:QUIT
echo [INFO] Saliendo...
goto END

:END
pause
