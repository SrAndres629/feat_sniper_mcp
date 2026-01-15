@echo off
set "EDITOR=C:\Program Files\LiteFinance MT5 Terminal\MetaEditor64.exe"
set "MT5_PATH=C:\Users\acord\AppData\Roaming\MetaQuotes\Terminal\065434634B76DD288A1DDF20131E8DDB\MQL5"
set "LOG_FILE=%~dp0\compile_log.txt"

echo [COMPILE] Target: FEAT_Visualizer.mq5
if not exist "%EDITOR%" (
    echo [ERROR] MetaEditor not found at: %EDITOR%
    exit /b 1
)

echo [EXEC] Compiling...
"%EDITOR%" /compile:"%MT5_PATH%\Indicators\FEAT\FEAT_Visualizer.mq5" /log:"%LOG_FILE%"

if exist "%LOG_FILE%" (
    echo [RESULT] Compilation Log:
    type "%LOG_FILE%"
) else (
    echo [ERROR] No log file generated.
)

pause
