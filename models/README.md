# 人脸检测系统模型库

这个目录包含人脸检测系统使用的各种模型。

## 模型列表

### YOLOv7-Face

- 位置: `yolov7-face/`
- 功能: 高精度的实时人脸检测
- 依赖: PyTorch, OpenCV
- 主要文件:
  - `weights/yolov7-face.pt`: 预训练的YOLOv7-Face模型权重
  - `face_db.py`: 人脸数据库管理功能

### 情感检测 (Emotion Detection)

- 位置: `emotion_detection/`
- 功能: 基于ViT (Vision Transformer) 的情感识别
- 依赖: PyTorch, torchvision, timm
- 主要文件:
  - `emotion_model.py`: 情感检测模型实现
  - `weights/emotion_vit_model.pth`: 预训练的情感识别模型权重（需要单独下载）
- 支持的情感类别:
  - 愤怒 (Angry)
  - 厌恶 (Disgust)
  - 恐惧 (Fear)
  - 高兴 (Happy)
  - 悲伤 (Sad)
  - 惊讶 (Surprise)
  - 中性 (Neutral)

### 疲劳检测 (Fatigue Detection)

- 位置: `fatigue_detection/`
- 功能: 基于YOLOv8的疲劳状态检测
- 依赖: OpenCV, ultralytics
- 主要文件:
  - `fatigue_model.py`: 疲劳检测模型实现
  - `weights/yolov8n.pt`: YOLOv8模型权重（会自动下载）
- 检测的疲劳指标:
  - 眼睛闭合时间过长
  - 打哈欠
  - 低头姿势
  - 综合疲劳级别（0-100%）

### 模型集成

- 位置: `integration.py`
- 功能: 整合所有检测模型，提供统一的接口
- 主要功能:
  - 同时运行多个检测模型
  - 支持帧跳过以提高性能
  - 性能统计和监控
  - 可配置的模型启用/禁用选项

## 使用方法

### 模型集成使用示例

```python
from models.integration import DetectionManager

# 创建配置
config = {
    'enable_emotion_detection': True,  # 启用情感检测
    'enable_fatigue_detection': True,  # 启用疲劳检测
    'enable_frame_skip': True,         # 启用帧跳过
    'frame_skip': 2                    # 每处理1帧后跳过2帧
}

# 初始化检测管理器
detection_manager = DetectionManager(config)

# 在视频帧上运行检测
processed_frame, results = detection_manager.process_frame(frame)

# 获取检测结果
emotions = results['emotions']
fatigue_info = results['fatigue']

# 更新配置
detection_manager.update_config({'enable_frame_skip': False})
```

### 单独使用情感检测

```python
from models.emotion_detection.emotion_model import EmotionDetector

# 初始化情感检测器
emotion_detector = EmotionDetector()

# 在图像上检测情感
annotated_frame, emotions = emotion_detector.detect_emotion(frame)
```

### 单独使用疲劳检测

```python
from models.fatigue_detection.fatigue_model import FatigueDetector

# 初始化疲劳检测器
fatigue_detector = FatigueDetector()

# 在图像上检测疲劳
annotated_frame, is_fatigued, state = fatigue_detector.detect_fatigue(frame)
```

## 模型权重下载

某些模型权重需要单独下载：

- 情感检测模型权重 (emotion_vit_model.pth): 需要放置在 `emotion_detection/weights/` 目录下
- YOLOv8模型权重: 首次运行时会自动下载

## 系统配置

可以通过编辑 `backend/app.py` 中的 `detection_settings` 变量来控制各种检测参数。

## 模型目录结构

```
models/
├── yolov7-face/         # YOLOv7人脸检测模型
├── emotion_detection/   # 情感检测模型
├── fatigue_detection/   # 疲劳检测模型
└── integration.py       # 模型集成工具
```

## 模型说明

### YOLOv7-Face

使用YOLOv7架构的人脸检测模型，能够准确定位图像或视频中的人脸位置。

- 模型文件: `yolov7-face/weights/yolov7-face.pt`
- 主要功能: 实时人脸检测、定位和识别
- 数据库: 使用SQLite保存人脸特征向量和ID

### 情感检测模型

基于Vision Transformer (ViT)的情感识别模型，能够识别7种基本情感状态。

- 模型文件: `emotion_detection/weights/emotion_vit_model.pth`
- 支持的情感类别: 愤怒、厌恶、恐惧、高兴、悲伤、惊讶、平静
- 输入要求: 裁剪好的人脸图像
- 输出格式: 情感类别和置信度

### 疲劳检测模型

基于YOLOv8的疲劳检测模型，通过监测眼睛状态来判断用户疲劳程度。

- 模型文件: `fatigue_detection/weights/yolov8n.pt`
- 主要功能: 实时检测眼睛状态，计算疲劳程度
- 特点: 能够检测闭眼时间，并给出疲劳警告

## 使用方法

系统默认仅启用YOLOv7-Face人脸检测功能，情感检测和疲劳检测功能默认关闭。如需启用这些功能，请在设置面板中打开相应开关。

### 启用情感检测

1. 打开设置面板
2. 在"增强功能"区域中，勾选"启用情感检测"
3. 点击"保存设置"按钮

启用后，系统将在检测到的人脸上方显示情感状态和置信度。

### 启用疲劳检测

1. 打开设置面板
2. 在"增强功能"区域中，勾选"启用疲劳检测"
3. 点击"保存设置"按钮

启用后，系统将在画面上方显示疲劳警告和疲劳度指示器，当用户闭眼时间超过阈值将触发警告。

### 帧跳过功能

为了提高系统性能，默认启用了帧跳过功能。系统会每隔N帧处理一次人脸检测，但会在每一帧都显示检测结果，以保持视觉连贯性。

可以通过以下方式调整:

1. 打开设置面板
2. 在"增强功能"区域中，调整"帧跳过数量"滑块
3. 数值越高性能越好，但可能降低检测准确性
4. 设置为0可以禁用帧跳过

## 注意事项

1. 启用情感检测和疲劳检测会增加系统负担，可能导致帧率下降
2. 建议在性能较好的设备上同时启用多个功能
3. 如系统卡顿，可以尝试增大帧跳过数量或关闭部分功能
4. 疲劳检测需要较好的光照条件才能准确检测眼睛状态 