#!/bin/bash
# Visual Tension 半自动图像标注工具 - 启动脚本（macOS/Linux）

# 显示标题
echo "===================================================="
echo "   Visual Tension 半自动图像标注工具"
echo "===================================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 Python3"
    echo "请先安装 Python 3.8 或更高版本"
    echo "下载地址：https://www.python.org/downloads/"
    exit 1
fi

# 显示Python版本
echo "检测到Python版本："
python3 --version
echo ""

# 检查主程序文件
if [ ! -f "labeling_tool.py" ]; then
    echo "错误：找不到 labeling_tool.py 文件"
    echo "请确保在正确的目录下运行此脚本"
    exit 1
fi

# 检查是否需要安装依赖
if [ ! -d "venv" ] && [ -f "requirements.txt" ]; then
    echo "首次运行，正在检查依赖..."
    python3 -m pip install --user -r requirements.txt
    echo ""
fi

echo "正在启动程序..."
echo ""

# 启动主程序
python3 labeling_tool.py

# 程序退出后的提示
echo ""
echo "程序已退出"
