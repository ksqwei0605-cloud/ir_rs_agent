@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 尚未安装本地环境，请先双击“安装本地环境.bat”。
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python run_local.py
if errorlevel 1 pause
