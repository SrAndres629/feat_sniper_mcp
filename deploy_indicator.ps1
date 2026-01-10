$ErrorActionPreference = "Stop"

# --- Configuration ---
$SourceDir = "c:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp\FEAT_Sniper_Master_Core"
$MT5DataDir = "C:\Users\acord\AppData\Roaming\MetaQuotes\Terminal\065434634B76DD288A1DDF20131E8DDB\MQL5"
$CompilerPath = "C:\Program Files\LiteFinance MT5 Terminal\metaeditor64.exe"

$DestIndicators = "$MT5DataDir\Indicators\FEAT_Sniper"
$DestIncludes = "$MT5DataDir\Include\UnifiedModel"

# --- 1. Sync Include Files ---
Write-Host "1. Syncing Include Files..." -ForegroundColor Cyan
if (!(Test-Path $DestIncludes)) { New-Item -ItemType Directory -Path $DestIncludes -Force | Out-Null }
Copy-Item "$SourceDir\Include\UnifiedModel\*" -Destination $DestIncludes -Recurse -Force
Write-Host "   Includes updated." -ForegroundColor Gray

# --- Function to Compile and Deploy ---
function Invoke-IndicatorDeployment {
    param (
        [string]$FileName
    )
    
    $Source = "$SourceDir\$FileName"
    $Compiled = "$SourceDir\$($FileName.Replace('.mq5', '.ex5'))"
    $Log = "$SourceDir\compile_$($FileName.Replace('.mq5', '')).log"
    
    Write-Host "Compiling $FileName..." -ForegroundColor Cyan
    if (Test-Path $Log) { Remove-Item $Log }
    
    $CompileArgs = "/compile:`"$Source`" /log:`"$Log`""
    Start-Process -FilePath $CompilerPath -ArgumentList $CompileArgs -Wait -NoNewWindow
    
    if (Test-Path $Log) {
        $LogContent = Get-Content $Log -Encoding Unicode -Raw
        if ($LogContent -match "0 errors") {
            Write-Host "   SUCCESS: $FileName" -ForegroundColor Green
            
            # Deploy
            if (!(Test-Path $DestIndicators)) { New-Item -ItemType Directory -Path $DestIndicators -Force | Out-Null }
            Copy-Item $Compiled -Destination "$DestIndicators\$($FileName.Replace('.mq5', '.ex5'))" -Force
            
            # Log Timestamp
            $Ts = (Get-Item "$DestIndicators\$($FileName.Replace('.mq5', '.ex5'))").LastWriteTime
            Write-Host "   Deployed to MT5 ($Ts)" -ForegroundColor Gray
        }
        else {
            Write-Host "   FAILED: $FileName" -ForegroundColor Red
            Write-Host "   --- Log Output ---"
            Write-Host $LogContent
            exit 1
        }
    }
    else {
        Write-Host "Error: No log file for $FileName" -ForegroundColor Red
        exit 1
    }
}

# --- 2. Compile Main Code ---
Invoke-IndicatorDeployment -FileName "UnifiedModel_Main.mq5"

# --- 3. Compile PVP Module ---
Invoke-IndicatorDeployment -FileName "InstitutionalPVP.mq5"

# --- 4. Compile Diagnostic Ping ---
Invoke-IndicatorDeployment -FileName "Ping.mq5"

Write-Host "--- ALL DEPLOYMENTS COMPLETE ---" -ForegroundColor Yellow
Write-Host "PLEASE REFRESH MT5 NAVIGATOR." -ForegroundColor Yellow
