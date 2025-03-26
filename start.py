#!/usr/bin/env python
import os
import sys
import subprocess
import webbrowser
import time
from colorama import init, Fore, Style

# 初始化colorama
init()

def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     基于yolov7的人脸情绪疲劳检测系统                        ║
    ║                                                           ║
    ║     版本: 1.0.0                                           ║
    ║     作者: 王星裕                                          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(Fore.CYAN + banner + Style.RESET_ALL)

def check_requirements():
    print(Fore.YELLOW + "检查环境依赖..." + Style.RESET_ALL)
    
    # 检查Python版本
    python_version = sys.version.split()[0]
    print(f"Python版本: {python_version}")
    
    # 检查后端依赖
    backend_req_path = os.path.join("backend", "requirements.txt")
    if os.path.exists(backend_req_path):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", backend_req_path])
            print(Fore.GREEN + "后端依赖安装完成" + Style.RESET_ALL)
        except subprocess.CalledProcessError:
            print(Fore.RED + "安装后端依赖失败，请手动安装" + Style.RESET_ALL)
            return False
    else:
        print(Fore.RED + f"找不到后端依赖文件: {backend_req_path}" + Style.RESET_ALL)
        return False
    
    # 检查模型文件
    yolov7_face_pt = os.path.join("models", "yolov7-face", "weights", "yolov7-face.pt")
    if not os.path.exists(yolov7_face_pt):
        print(Fore.RED + f"找不到模型文件: {yolov7_face_pt}" + Style.RESET_ALL)
        return False
    
    print(Fore.GREEN + "环境检查完成" + Style.RESET_ALL)
    return True

def start_server():
    print(Fore.YELLOW + "启动服务器..." + Style.RESET_ALL)
    
    # 启动Flask应用
    os.environ['FLASK_APP'] = 'backend/app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        server_process = subprocess.Popen([sys.executable, "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"])
        print(Fore.GREEN + "服务器已启动: http://localhost:5000" + Style.RESET_ALL)
        
        # 等待服务器启动
        time.sleep(2)
        
        # 打开浏览器
        print(Fore.YELLOW + "正在打开浏览器..." + Style.RESET_ALL)
        webbrowser.open('http://localhost:5000')
        
        return server_process
    except Exception as e:
        print(Fore.RED + f"启动服务器失败: {str(e)}" + Style.RESET_ALL)
        return None

def main():
    print_banner()
    
    # 检查需求
    if not check_requirements():
        print(Fore.RED + "环境检查失败，无法启动系统" + Style.RESET_ALL)
        return
    
    # 启动服务器
    server_process = start_server()
    if not server_process:
        return
    
    # 等待用户退出
    print(Fore.CYAN + "\n系统已启动! 按 Ctrl+C 退出" + Style.RESET_ALL)
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n正在关闭服务..." + Style.RESET_ALL)
        server_process.terminate()
        server_process.wait()
        print(Fore.GREEN + "系统已安全退出" + Style.RESET_ALL)

if __name__ == "__main__":
    main() 
