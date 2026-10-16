@echo off
title GNSS dashboard self-test
cd /d "%~dp0"
call setup_env.bat
if errorlevel 1 goto end
".venv\Scripts\python.exe" tests\selftest.py
:end
echo.
pause
