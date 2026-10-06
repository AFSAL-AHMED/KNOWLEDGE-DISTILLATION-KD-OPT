# =========================================================
#  MiniLLM Knowledge Distillation -- PowerShell Run Script
#  Run with:  .\run.ps1
# =========================================================

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  Launching MiniLLM Knowledge Distillation (main.py)    " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

if (Test-Path "venv\Scripts\python.exe") {
    Write-Host "[INFO] Virtual environment found. Running within venv..." -ForegroundColor Green
    & "venv\Scripts\python.exe" main.py
} else {
    Write-Host "[INFO] Running with Python..." -ForegroundColor Green
    python main.py
}
