# =========================================================
#  Launch Knowledge Distillation Web Studio -- run_web.ps1
#  Run with:  .\run_web.ps1
# =========================================================

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "  Launching MiniLLM Web Studio (Frontend + Inference API) " -ForegroundColor Cyan
Write-Host "  Port: http://localhost:7860                             " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

if (Test-Path "venv\Scripts\python.exe") {
    $PYTHON = "venv\Scripts\python.exe"
} else {
    $PYTHON = "python"
}

Write-Host "`n[INFO] Starting web server ...`n" -ForegroundColor Green
& $PYTHON server.py
