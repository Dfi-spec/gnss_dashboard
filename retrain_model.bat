@echo off
title Retrain the model (ml_model.py)
cd /d "%~dp0"
call setup_env.bat
if errorlevel 1 goto end
echo.
echo This runs your ml_model.py. It rebuilds signal_dataset.csv and signal_model.joblib.
echo The current files are copied to the "backup" folder first.
echo.
pause
if not exist backup mkdir backup
copy /y signal_model.joblib backup\signal_model.joblib >nul
copy /y signal_dataset.csv backup\signal_dataset.csv >nul
".venv\Scripts\python.exe" ml_model.py
:end
echo.
pause
