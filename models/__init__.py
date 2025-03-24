#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检测模型包
此模块导入所有可用的检测模型
"""

import os
import sys

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"添加模型路径到系统路径: {current_dir}")

# 导入集成接口
from .integration import DetectionManager, import_face_detection

# 尝试导入各个检测模型
try:
    from .emotion_detection.emotion_model import EmotionDetector
    _has_emotion_detection = True
    print("成功导入情感检测模型")
except ImportError as e:
    _has_emotion_detection = False
    print(f"无法导入情感检测模型: {str(e)}")

try:
    from .fatigue_detection.fatigue_model import FatigueDetector
    _has_fatigue_detection = True
    print("成功导入疲劳检测模型")
except ImportError as e:
    _has_fatigue_detection = False
    print(f"无法导入疲劳检测模型: {str(e)}")

# 导出的接口
__all__ = ['DetectionManager', 'import_face_detection']

# 如果可用，添加到导出列表
if _has_emotion_detection:
    __all__.append('EmotionDetector')

if _has_fatigue_detection:
    __all__.append('FatigueDetector')

# 输出可用检测器信息
print(f"可用检测器: 情感检测({'✓' if _has_emotion_detection else '✗'}), "
      f"疲劳检测({'✓' if _has_fatigue_detection else '✗'})") 