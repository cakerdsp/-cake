"""
依赖包安装脚本
"""
import subprocess
import sys

def install_requirements():
    """安装所需的Python包"""
    packages = [
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    print("正在安装依赖包...")
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"安装 {package} 失败: {e}")
            return False
    
    print("所有依赖包安装完成！")
    return True

if __name__ == "__main__":
    install_requirements()
