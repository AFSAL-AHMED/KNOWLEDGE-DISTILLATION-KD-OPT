@echo off
REM =========================================================
REM  MiniLLM Knowledge Distillation -- Quick Setup Script
REM  Run once:   setup.bat
REM  Then run:   run.bat
REM =========================================================

echo [1/3] Creating virtual environment...
python -m venv venv

echo [2/3] Activating venv and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo [3/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Done! Virtual environment is ready.
echo To run the project:  run.bat
