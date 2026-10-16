@echo off
rem Creates the .venv folder and installs the packages the first time. Called by the other .bat files.
cd /d "%~dp0"

set "PYCMD="
py -3.13 --version >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3.13"
if not defined PYCMD (
  py -3 --version >nul 2>nul
  if not errorlevel 1 set "PYCMD=py -3"
)
if not defined PYCMD (
  python --version >nul 2>nul
  if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD goto nopython

if exist ".venv\Scripts\python.exe" goto havevenv
echo Creating the virtual environment (first run only)...
%PYCMD% -m venv .venv
if errorlevel 1 goto fail

:havevenv
if exist ".venv\.installed" goto done
echo Installing the packages from requirements.txt (first run only, needs internet)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo ok> ".venv\.installed"

:done
exit /b 0

:nopython
echo.
echo Python was not found.
echo Install Python 3.13 from https://www.python.org/downloads/  and tick "Add python.exe to PATH",
echo then run this file again.
exit /b 1

:fail
echo.
echo Setup failed. Read the messages above. If the problem is a package install, check your internet
echo connection and run this file again.
exit /b 1
