#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Tension 半自动图像标注工具 - 安装脚本
自动检查环境并安装依赖
"""

import sys
import subprocess
import importlib
import os

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"   当前版本：Python {version.major}.{version.minor}.{version.micro}")
        print("   请升级Python后重试")
        sys.exit(1)
    else:
        print(f"✅ Python版本合适：{version.major}.{version.minor}.{version.micro}")

def check_pip():
    """检查pip是否可用"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True, check=True)
        print("✅ pip可用")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip不可用，请先安装pip")
        return False

def install_requirements():
    """安装依赖库"""
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print("❌ 找不到requirements.txt文件")
        return False

    print("\n🔄 开始安装依赖库...")
    try:
        # 升级pip
        print("   升级pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                      check=True)

        # 安装依赖
        print("   安装项目依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file],
                      check=True)
        print("✅ 依赖库安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败：{e}")
        return False

def check_dependencies():
    """检查关键依赖是否安装成功"""
    required_modules = {
        'customtkinter': 'CustomTkinter UI框架',
        'PIL': 'Pillow图像处理库',
        'requests': 'HTTP请求库',
        'urllib3': 'HTTP库'
    }

    print("\n🔍 检查依赖库安装状态...")
    all_good = True

    for module, description in required_modules.items():
        try:
            importlib.import_module(module)
            print(f"✅ {description} - 已安装")
        except ImportError:
            print(f"❌ {description} - 未安装")
            all_good = False

    return all_good

def check_tkinter():
    """检查tkinter是否可用"""
    try:
        import tkinter
        print("✅ tkinter GUI库 - 可用")
        return True
    except ImportError:
        print("❌ tkinter不可用，请安装python3-tk")
        print("   Ubuntu/Debian: sudo apt-get install python3-tk")
        print("   CentOS/RHEL: sudo yum install tkinter")
        print("   macOS: tkinter通常随Python安装")
        return False

def create_startup_scripts():
    """创建启动脚本"""
    # Windows批处理文件
    bat_content = """@echo off
echo Starting Visual Tension Labeling Tool...
python labeling_tool.py
pause"""

    try:
        with open("start.bat", "w", encoding="utf-8") as f:
            f.write(bat_content)
        print("✅ 创建Windows启动脚本：start.bat")
    except:
        pass

    # 确保run.sh有执行权限
    if os.path.exists("run.sh"):
        try:
            os.chmod("run.sh", 0o755)
            print("✅ 设置Unix启动脚本权限：run.sh")
        except:
            pass

def main():
    """主函数"""
    print("=" * 60)
    print("Visual Tension 半自动图像标注工具 - 环境配置")
    print("=" * 60)

    # 检查Python版本
    check_python_version()

    # 检查pip
    if not check_pip():
        sys.exit(1)

    # 检查tkinter
    if not check_tkinter():
        print("\n⚠️  警告：tkinter不可用，程序可能无法启动")

    # 安装依赖
    if not install_requirements():
        sys.exit(1)

    # 验证安装
    if not check_dependencies():
        print("\n❌ 部分依赖库安装失败，请手动安装")
        sys.exit(1)

    # 创建启动脚本
    create_startup_scripts()

    print("\n" + "=" * 60)
    print("🎉 安装完成！")
    print("=" * 60)
    print("\n启动方式：")
    print("  方法1：python labeling_tool.py")
    if os.path.exists("run.sh"):
        print("  方法2：./run.sh (macOS/Linux)")
    if os.path.exists("start.bat"):
        print("  方法3：双击 start.bat (Windows)")

    print("\n使用前请：")
    print("  1. 准备好待标注的图像文件夹")
    print("  2. 如需AI标注，准备OpenAI API Key")
    print("  3. 阅读 README.md 了解详细使用方法")
    print("\n享受使用！")

if __name__ == "__main__":
    main()
