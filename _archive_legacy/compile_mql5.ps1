$metaEditor = "C:\Program Files\LiteFinance MT5 Terminal\MetaEditor64.exe"
$appDataPath = "C:\Users\acord\AppData\Roaming\MetaQuotes\Terminal\065434634B76DD288A1DDF20131E8DDB"

$targets = @(
    "$appDataPath\MQL5\Indicators\FEAT_Sniper\UnifiedModel_Main.mq5",
    "$appDataPath\MQL5\Indicators\FEAT_Sniper\InstitutionalPVP.mq5",
    "$appDataPath\MQL5\Indicators\FEAT_Sniper\Ping.mq5"
)

foreach ($targetFile in $targets) {
    $logFile = "$targetFile.log"
    Write-Host "Compiling $targetFile..."
    Start-Process -FilePath $metaEditor -ArgumentList "/compile:$targetFile", "/log:$logFile" -Wait

    if (Test-Path $logFile) {
        $content = Get-Content $logFile -Encoding Unicode
        if ($content -match "0 errors") {
            Write-Host "Success: $(Split-Path $targetFile -Leaf) compiled with 0 errors."
        }
        else {
            Write-Host "Errors found in $(Split-Path $targetFile -Leaf). See log."
        }
    }
}
