@echo off
chcp 65001 >nul
echo ====================================================
echo    Visual Tension 半自动图像标注工具
echo ====================================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 Python，请先安装 Python 3.8+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 显示Python版本
echo 检测到Python版本：
python --version

:: 检查主程序文件
if not exist "labeling_tool.py" (
    echo 错误：找不到 labeling_tool.py 文件
    echo 请确保在正确的目录下运行此脚本
    pause
    exit /b 1
)

echo.
echo 正在启动程序...
echo.

:: 启动主程序
python labeling_tool.py

:: 程序结束后暂停
echo.
echo 程序已退出
pause
