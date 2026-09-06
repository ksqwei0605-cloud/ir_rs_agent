@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo IR-RS-Agent 本地环境安装
echo ========================================

if exist ".venv\Scripts\python.exe" (
  echo 已发现现有虚拟环境，将继续检查和补齐依赖。
) else (
  echo 正在创建 Python 3.12 虚拟环境...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo 未找到 Python 3.12，请先安装 Python 3.12 64位版本。
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo 正在安装支持 NVIDIA 显卡的 PyTorch...
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :failed

echo 正在安装 Agent、网页和检测依赖...
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
if errorlevel 1 goto :failed

python check_environment.py
if errorlevel 1 goto :failed

echo.
echo 安装完成。下一步填写 .env 中的 DASHSCOPE_API_KEY，随后双击“启动IR-RS-Agent.bat”。
pause
exit /b 0

:failed
echo.
echo 安装失败，请保留本窗口中的错误信息。
pause
exit /b 1
