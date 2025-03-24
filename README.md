# YOLOv7-Face 人脸检测系统

基于YOLOv7-face的实时人脸检测与管理系统，提供了Web界面，支持摄像头实时检测、人脸识别与数据库管理功能。

## 系统功能

- **用户管理**：支持登录和注册功能
- **实时人脸检测**：基于YOLOv7-face模型的实时人脸检测
- **人脸数据库管理**：可以查看、删除数据库中的人脸记录
- **检测数据统计**：显示检测到的人脸数量和时间统计

## 系统架构

- **前端**：HTML5 + CSS3 + JavaScript，使用Bootstrap 5框架
- **后端**：Python Flask提供API服务
- **模型**：YOLOv7-face用于人脸检测
- **数据库**：SQLite存储人脸编码和用户信息

## 目录结构

```
Human_Detect/
│
├── backend/                 # 后端Flask应用
│   ├── app.py               # 主应用文件
│   └── requirements.txt     # 后端依赖
│
├── frontend/                # 前端文件
│   ├── static/              # 静态资源
│   │   ├── css/             # CSS样式
│   │   ├── js/              # JavaScript脚本
│   │   └── img/             # 图片资源
│   └── templates/           # HTML模板
│
├── models/                  # 模型目录
│   └── yolov7-face/         # YOLOv7-face模型
│       ├── weights/         # 模型权重
│       │   └── yolov7-face.pt  # 模型文件
│       └── faces.db         # 人脸数据库
│
├── start.py                 # 启动脚本
└── README.md                # 项目说明文档
```

## 依赖要求

- Python 3.8+
- PyTorch 1.8+
- OpenCV
- Flask
- Flask-CORS
- face-recognition
- dlib

## 快速开始

1. 确保已安装Python 3.8或更高版本
2. 运行启动脚本：

```bash
python start.py
```

启动脚本会自动：
- 检查环境依赖并安装所需包
- 启动Flask后端服务
- 打开浏览器访问系统

## 默认账号

系统默认账号：
- 用户名：users
- 密码：users123

## 贡献者

- AI团队

## 许可证

本项目使用MIT许可证 