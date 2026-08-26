@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0env\python.exe" (
  "%~dp0env\python.exe" "%~dp0pyinstaller_packaging\sem_grain_web_app.py"
) else (
  echo Missing portable Python environment: %~dp0env\python.exe
  echo Put the packed conda environment in the env folder.
  pause
)
