@echo off
title GNSS Jamming Detection Analyzer
cd /d "%~dp0"
call setup_env.bat
if errorlevel 1 goto end

if not defined GNSS_PORT set "GNSS_PORT=5000"
echo.
echo Starting the dashboard on http://127.0.0.1:%GNSS_PORT%  (close this window or press Ctrl+C to stop)
start "" cmd /c "ping -n 5 127.0.0.1 >nul & start http://127.0.0.1:%GNSS_PORT%"
".venv\Scripts\python.exe" server.py

:end
echo.
pause
