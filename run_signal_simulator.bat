@echo off
title signal_simulator.py (your original script)
cd /d "%~dp0"
call setup_env.bat
if errorlevel 1 goto end
".venv\Scripts\python.exe" signal_simulator.py
:end
echo.
pause
