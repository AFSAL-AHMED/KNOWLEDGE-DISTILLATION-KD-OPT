# =========================================================
#  Run In-Family Experiment -- run_infamily.ps1
#  Run with:  .\run_infamily.ps1
# =========================================================

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  In-Family vs Cross-Family Distillation Experiment      " -ForegroundColor Cyan
Write-Host "  Testing OPT-125m and Pythia-70m                        " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

if (Test-Path "venv\Scripts\python.exe") {
    $PYTHON = "venv\Scripts\python.exe"
    Write-Host "[INFO] Using virtual environment Python: $PYTHON" -ForegroundColor Green
} else {
    $PYTHON = "python"
    Write-Host "[INFO] Using system Python." -ForegroundColor Yellow
}

Write-Host "`n[INFO] Launching infamily_compare.py ...`n" -ForegroundColor Green
& $PYTHON infamily_compare.py
