param(
    [double]$StartThreshold = 0.60,
    [double]$EndThreshold = 0.90,
    [double]$Step = 0.05,
    [switch]$SkipSweep,
    [ValidateSet("support", "devdocs", "compliance")]
    [string]$Domain = "support",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $ProjectRoot
try {
    if ($SkipSweep) {
        & $PythonExe -m src.main run-all --domain $Domain --skip-sweep
    }
    else {
        & $PythonExe -m src.main run-all --domain $Domain --start-threshold $StartThreshold --end-threshold $EndThreshold --step $Step
    }

    if ($LASTEXITCODE -ne 0) {
        throw "run-all failed with exit code $LASTEXITCODE"
    }

    Write-Host "Artifacts prepared in logs/."
    Write-Host "Launch dashboard with: streamlit run dashboard/app.py"
}
finally {
    Pop-Location
}
