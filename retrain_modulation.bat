@echo off
title Retrain the modulation classifier (train_modulation.py)
cd /d "%~dp0"
call setup_env.bat
if errorlevel 1 goto end
echo.
echo This rebuilds modulation_dataset.csv, modulation_model.joblib and modulation_metrics.json (about 1 minute).
echo The current files are copied to the "backup" folder first. The jamming analyzer files are NOT touched.
echo.
pause
if not exist backup mkdir backup
copy /y modulation_model.joblib backup\modulation_model.joblib >nul
copy /y modulation_dataset.csv backup\modulation_dataset.csv >nul
".venv\Scripts\python.exe" train_modulation.py
:end
echo.
pause
