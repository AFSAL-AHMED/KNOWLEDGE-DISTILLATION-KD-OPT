# =========================================================
#  Model Comparison Run Script  --  run_compare.ps1
#  Run with:  .\run_compare.ps1
# =========================================================

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  Model Architecture Comparison (compare.py)            " -ForegroundColor Cyan
Write-Host "  Exp A: Fixed Teacher OPT-350m, Vary Student           " -ForegroundColor Cyan
Write-Host "  Exp B: Fixed Student OPT-125m, Vary Teacher           " -ForegroundColor Cyan
Write-Host "  Eval : DollyEval | SelfInst | VicunaEval              " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

if (Test-Path "venv\Scripts\python.exe") {
    $PYTHON = "venv\Scripts\python.exe"
    Write-Host "[INFO] Using virtual environment." -ForegroundColor Green
} else {
    $PYTHON = "python"
    Write-Host "[INFO] Using system Python." -ForegroundColor Yellow
}

Write-Host "`n[INFO] Ensuring rouge-score is installed ..." -ForegroundColor Yellow
& $PYTHON -m pip install rouge-score --quiet

Write-Host "`n[INFO] Launching compare.py ...`n" -ForegroundColor Green
& $PYTHON compare.py
