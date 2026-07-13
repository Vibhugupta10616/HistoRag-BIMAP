# Run H3 exp01 linear probe for all three encoders.
# Usage: pwsh hypotheses/H3_supervised_probe/run_h3_all_encoders.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runPy = Join-Path $scriptDir "exp01_linear_probe\run.py"

foreach ($encoder in @("uni2h", "conch", "clip-vitb16")) {
    Write-Host "=== H3 exp01 — encoder=$encoder ===" -ForegroundColor Cyan
    python $runPy --encoder $encoder
}
