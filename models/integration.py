#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检测模型集成模块
提供统一的接口来管理所有检测模型
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"添加集成模块路径到系统路径: {current_dir}")

# 尝试导入各个检测模型
try:
    from emotion_detection.emotion_model import EmotionDetector
    print("成功导入情感检测模型")
    _has_emotion_detection = True
except ImportError as e:
    print(f"无法导入情感检测模型: {str(e)}")
    _has_emotion_detection = False

try:
    from fatigue_detection.fatigue_model import FatigueDetector
    print("成功导入疲劳检测模型")
    _has_fatigue_detection = True
except ImportError as e:
    print(f"无法导入疲劳检测模型: {str(e)}")
    _has_fatigue_detection = False

def import_face_detection():
    """导入人脸检测模型"""
    try:
        from yolov7_face.detect import detect
        return detect
    except ImportError as e:
        print(f"导入人脸检测模型失败: {str(e)}")
        return None

class DetectionManager:
    """检测管理器，用于集成所有检测模型"""
    
    def __init__(self, config=None):
        """
        初始化检测管理器
        
        参数:
            config: 配置字典，包含各个模型的配置
        """
        self.config = config or {}
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"检测管理器使用设备: {self.device}")
        
        # 加载情感检测模型
        self.emotion_enabled = self.config.get('enable_emotion_detection', False)
        if self.emotion_enabled:
            print("启用情感检测模型")
            try:
                emotion_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                'emotion_detection', 'weights', 'emotion_vit_model.pth')
                self.emotion_detector = EmotionDetector(model_path=emotion_model_path)
                print("情感检测模型加载成功")
            except Exception as e:
                print(f"加载情感检测模型失败: {str(e)}")
                self.emotion_detector = None
                self.emotion_enabled = False
        else:
            self.emotion_detector = None
            
        # 加载疲劳检测模型
        self.fatigue_enabled = self.config.get('enable_fatigue_detection', False)
        if self.fatigue_enabled:
            print("启用疲劳检测模型")
            try:
                fatigue_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                'fatigue_detection', 'weights', 'yolov8n.pt')
                self.fatigue_detector = FatigueDetector(model_path=fatigue_model_path)
                print("疲劳检测模型加载成功")
            except Exception as e:
                print(f"加载疲劳检测模型失败: {str(e)}")
                self.fatigue_detector = None
                self.fatigue_enabled = False
        else:
            self.fatigue_detector = None
        
        # 初始化变量
        self.last_update_time = time.time()
        self.frame_skip = self.config.get('frame_skip', 2)
        self.enable_frame_skip = self.config.get('enable_frame_skip', True)
        self.frame_count = 0
        
        # 存储最新的检测结果
        self.latest_results = {
            'emotions': [],
            'fatigue': {
                'is_fatigued': False,
                'fatigue_level': 0,
                'state': {}
            }
        }
        
        # 跟踪统计信息
        self.performance_stats = {
            'emotion_detection_time': 0,
            'fatigue_detection_time': 0,
            'total_time': 0,
            'frames_processed': 0
        }
        
        # 加载人脸检测模型
        self.face_detector = import_face_detection()
        if self.face_detector:
            print("人脸检测模型加载成功")
        else:
            print("人脸检测模型加载失败")
        
        print("检测管理器初始化完成")
    
    def process_frame(self, frame):
        """
        处理单帧，运行所有启用的检测模型
        
        参数:
            frame: 输入的视频帧
            
        返回:
            处理后的帧和检测结果
        """
        if frame is None or frame.size == 0:
            return frame, self.latest_results
        
        # 增加帧计数
        self.frame_count += 1
        
        # 决定是否处理此帧
        should_process = True
        if self.enable_frame_skip:
            should_process = (self.frame_count % (self.frame_skip + 1) == 1)
        
        results = self.latest_results.copy()
        processed_frame = frame.copy()
        start_time = time.time()
        
        try:
            # 只有在应该处理该帧时执行检测
            if should_process:
                # 运行情感检测
                if self.emotion_enabled and self.emotion_detector:
                    try:
                        emotion_start = time.time()
                        emotion_frame, emotions = self.emotion_detector.detect_emotion(processed_frame)
                        emotion_time = time.time() - emotion_start
                        
                        processed_frame = emotion_frame
                        results['emotions'] = emotions
                        
                        # 更新性能统计
                        self.performance_stats['emotion_detection_time'] += emotion_time
                    except Exception as e:
                        print(f"情感检测出错: {str(e)}")
                
                # 运行疲劳检测
                if self.fatigue_enabled and self.fatigue_detector:
                    try:
                        fatigue_start = time.time()
                        fatigue_frame, is_fatigued, state = self.fatigue_detector.detect_fatigue(processed_frame)
                        fatigue_time = time.time() - fatigue_start
                        
                        processed_frame = fatigue_frame
                        results['fatigue'] = {
                            'is_fatigued': is_fatigued,
                            'fatigue_level': state.get('fatigue_level', 0),
                            'state': state
                        }
                        
                        # 更新性能统计
                        self.performance_stats['fatigue_detection_time'] += fatigue_time
                    except Exception as e:
                        print(f"疲劳检测出错: {str(e)}")
                
                # 运行人脸检测
                if self.face_detector:
                    try:
                        faces = self.face_detector(processed_frame)
                        results['faces'] = faces
                        self.performance_stats['frames_processed'] += len(faces)
                    except Exception as e:
                        print(f"人脸检测出错: {str(e)}")
                
                # 更新最新结果
                self.latest_results = results
                self.performance_stats['frames_processed'] += 1
            else:
                # 对于跳过的帧，使用上一次的检测结果，但仍然在帧上绘制信息
                if self.emotion_enabled and results['emotions']:
                    # 绘制上一次检测到的情感
                    for emotion_data in results['emotions']:
                        if 'bbox' in emotion_data:
                            x, y, w, h = emotion_data['bbox']
                            # 绘制人脸框
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # 绘制情感标签
                            emotion_text = f"{emotion_data.get('emotion_zh', '')}/{emotion_data.get('emotion_en', '')}: {emotion_data.get('confidence', 0):.2f}"
                            cv2.putText(processed_frame, emotion_text, (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.fatigue_enabled and results['fatigue']['is_fatigued']:
                    # 绘制疲劳警告
                    level = results['fatigue']['fatigue_level']
                    cv2.putText(processed_frame, f"疲劳级别: {level}%", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 绘制进度条
                    bar_length = 200
                    filled_length = int(bar_length * level / 100)
                    
                    # 进度条背景
                    cv2.rectangle(processed_frame, (10, 160), (10 + bar_length, 180), (200, 200, 200), -1)
                    
                    # 填充部分 - 使用根据级别变化的颜色
                    if level < 30:
                        color = (0, 255, 0)  # 绿色
                    elif level < 70:
                        color = (0, 165, 255)  # 橙色
                    else:
                        color = (0, 0, 255)  # 红色
                        
                    if filled_length > 0:
                        cv2.rectangle(processed_frame, (10, 160), (10 + filled_length, 180), color, -1)
                    
                    # 进度条边框
                    cv2.rectangle(processed_frame, (10, 160), (10 + bar_length, 180), (0, 0, 0), 1)
                    
                    # 如果疲劳级别高，添加红色闪烁边框
                    if level > 70 and self.frame_count % 10 < 5:
                        h, w = processed_frame.shape[:2]
                        cv2.rectangle(processed_frame, (0, 0), (w, h), (0, 0, 255), 20)
            
            # 在帧上添加跳帧信息
            if self.enable_frame_skip:
                skip_text = "处理" if should_process else "跳过"
                cv2.putText(processed_frame, f"帧 {self.frame_count}: {skip_text}", 
                           (processed_frame.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
        
        # 计算总处理时间
        total_time = time.time() - start_time
        self.performance_stats['total_time'] += total_time
        
        # 在帧上添加性能信息
        if self.frame_count % 30 == 0:  # 每30帧更新一次性能信息
            avg_time = self.performance_stats['total_time'] / max(1, self.performance_stats['frames_processed'])
            fps = 1.0 / max(0.001, avg_time)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                       (processed_frame.shape[1] - 150, processed_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return processed_frame, results
    
    def update_config(self, new_config):
        """
        更新配置
        
        参数:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        
        # 更新情感检测设置
        self.emotion_enabled = self.config.get('enable_emotion_detection', self.emotion_enabled)
        if self.emotion_enabled and self.emotion_detector is None:
            try:
                emotion_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                'emotion_detection', 'weights', 'emotion_vit_model.pth')
                self.emotion_detector = EmotionDetector(model_path=emotion_model_path)
                print("情感检测模型加载成功")
            except Exception as e:
                print(f"加载情感检测模型失败: {str(e)}")
                self.emotion_detector = None
                self.emotion_enabled = False
        
        # 更新疲劳检测设置
        self.fatigue_enabled = self.config.get('enable_fatigue_detection', self.fatigue_enabled)
        if self.fatigue_enabled and self.fatigue_detector is None:
            try:
                fatigue_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                'fatigue_detection', 'weights', 'yolov8n.pt')
                self.fatigue_detector = FatigueDetector(model_path=fatigue_model_path)
                print("疲劳检测模型加载成功")
            except Exception as e:
                print(f"加载疲劳检测模型失败: {str(e)}")
                self.fatigue_detector = None
                self.fatigue_enabled = False
        
        # 更新帧跳过设置
        self.frame_skip = self.config.get('frame_skip', self.frame_skip)
        self.enable_frame_skip = self.config.get('enable_frame_skip', self.enable_frame_skip)
        
        # 更新人脸检测设置
        self.face_detector = import_face_detection()
        if self.face_detector:
            print("人脸检测模型加载成功")
        else:
            print("人脸检测模型加载失败")
        
        print(f"配置已更新: 情感检测={self.emotion_enabled}, 疲劳检测={self.fatigue_enabled}, 帧跳过={self.enable_frame_skip}({self.frame_skip})")
    
    def get_performance_stats(self):
        """
        获取性能统计信息
        
        返回:
            性能统计字典
        """
        stats = self.performance_stats.copy()
        if stats['frames_processed'] > 0:
            stats['avg_emotion_time'] = stats['emotion_detection_time'] / stats['frames_processed']
            stats['avg_fatigue_time'] = stats['fatigue_detection_time'] / stats['frames_processed']
            stats['avg_total_time'] = stats['total_time'] / stats['frames_processed']
            stats['avg_fps'] = 1.0 / max(0.001, stats['avg_total_time'])
        else:
            stats['avg_emotion_time'] = 0
            stats['avg_fatigue_time'] = 0
            stats['avg_total_time'] = 0
            stats['avg_fps'] = 0
        
        return stats

# 测试代码
if __name__ == "__main__":
    # 创建测试配置
    test_config = {
        'enable_emotion_detection': True,
        'enable_fatigue_detection': True,
        'enable_frame_skip': True,
        'frame_skip': 2
    }
    
    # 初始化检测管理器
    detection_manager = DetectionManager(test_config)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame, results = detection_manager.process_frame(frame)
        
        # 显示性能统计
        if detection_manager.frame_count % 100 == 0:
            stats = detection_manager.get_performance_stats()
            print(f"处理帧数: {stats['frames_processed']}, FPS: {stats['avg_fps']:.1f}")
        
        # 显示结果
        cv2.imshow('集成检测', processed_frame)
        
        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 按 'c' 键切换配置
        if cv2.waitKey(1) & 0xFF == ord('c'):
            test_config['enable_frame_skip'] = not test_config['enable_frame_skip']
            detection_manager.update_config(test_config)
            print(f"帧跳过已{'启用' if test_config['enable_frame_skip'] else '禁用'}")
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows() 