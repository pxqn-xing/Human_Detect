#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情感检测模型：使用预训练的ViT模型进行人脸情感分类
支持七种情感类别：生气、厌恶、恐惧、高兴、悲伤、惊讶和中性
"""

import os
import cv2
import torch
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image

# 定义情感类别（中英文对照）
EMOTION_LABELS = {
    0: {'en': 'Angry', 'zh': '生气'},
    1: {'en': 'Disgust', 'zh': '厌恶'},
    2: {'en': 'Fear', 'zh': '恐惧'},
    3: {'en': 'Happy', 'zh': '高兴'},
    4: {'en': 'Sad', 'zh': '悲伤'},
    5: {'en': 'Surprise', 'zh': '惊讶'},
    6: {'en': 'Neutral', 'zh': '中性'}
}

class EmotionDetector:
    def __init__(self, model_path=None, device=None):
        """
        初始化情感检测器
        
        Args:
            model_path: 模型权重文件路径，如果为None则使用默认路径
            device: 计算设备，如果为None则自动选择
        """
        # 设置设备
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"情感检测使用设备: {self.device}")
        
        # 定义图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 加载模型
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'weights', 'emotion_vit_model.pth')
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载预训练的情感检测模型"""
        try:
            print(f"加载情感检测模型: {model_path}")
            # 创建ViT模型
            self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
            
            # 修改最后一层全连接层以适应7种情感分类
            num_ftrs = self.model.head.in_features
            self.model.head = torch.nn.Linear(num_ftrs, 7)
            
            # 检查模型文件是否存在
            if os.path.exists(model_path):
                # 加载模型权重
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                print("情感检测模型加载成功")
            else:
                print(f"警告: 情感检测模型文件不存在: {model_path}")
                print("使用未训练的模型，预测结果将不准确")
                self.model = self.model.to(self.device)
                self.model.eval()
            
            return True
        except Exception as e:
            print(f"加载情感检测模型失败: {str(e)}")
            return False
    
    def detect_emotions(self, frame):
        """
        检测图像中的人脸并识别情感
        
        Args:
            frame: OpenCV格式的图像帧 (BGR)
            
        Returns:
            处理后的图像帧和检测到的情感列表
        """
        if frame is None:
            return frame, []
        
        # 复制帧以避免修改原始帧
        result_frame = frame.copy()
        
        # 转换为灰度图用于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # 存储检测到的情感结果
        emotions_results = []
        
        # 处理每个检测到的人脸
        for (x, y, w, h) in faces:
            try:
                # 提取人脸区域
                face_roi = frame[y:y+h, x:x+w]
                
                # 转换为PIL图像
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                
                # 应用图像转换
                input_tensor = self.transform(face_pil)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                # 情感预测
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    emotion_idx = predicted.item()
                    confidence_val = confidence.item()
                    
                    # 获取情感标签
                    emotion_en = EMOTION_LABELS[emotion_idx]['en']
                    emotion_zh = EMOTION_LABELS[emotion_idx]['zh']
                
                # 在图像上绘制人脸框和情感标签
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{emotion_zh} ({confidence_val:.2f})"
                cv2.putText(result_frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 添加到结果列表
                emotions_results.append({
                    'position': (x, y, w, h),
                    'emotion': {
                        'index': emotion_idx,
                        'en': emotion_en,
                        'zh': emotion_zh
                    },
                    'confidence': confidence_val
                })
                
            except Exception as e:
                print(f"处理人脸情感时出错: {str(e)}")
                # 如果处理失败，仍然绘制人脸框
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return result_frame, emotions_results
    
    def __call__(self, frame):
        """便捷调用方法，允许直接使用对象处理帧"""
        return self.detect_emotions(frame)

# 测试代码
if __name__ == "__main__":
    # 初始化情感检测器
    detector = EmotionDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 检测情感
        result_frame, emotions = detector(frame)
        
        # 显示结果
        cv2.imshow('Emotion Detection', result_frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows() 