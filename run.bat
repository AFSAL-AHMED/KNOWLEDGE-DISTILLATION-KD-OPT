@echo off
REM =========================================================
REM  MiniLLM Knowledge Distillation -- Run Script
REM =========================================================

if exist "venv\Scripts\python.exe" (
    echo [INFO] Virtual environment found. Running within venv...
    call venv\Scripts\activate.bat
    python main.py
) else (
    echo [INFO] Running with system Python...
    python main.py
)
pause
