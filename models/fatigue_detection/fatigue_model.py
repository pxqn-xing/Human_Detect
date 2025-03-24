import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

class FatigueDetector:
    def __init__(self, model_path=None):
        """
        初始化疲劳检测模型
        """
        print("初始化疲劳检测模型...")
        
        # 设置默认模型路径
        default_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       'weights', 'yolov8n.pt')
        
        # 使用提供的模型路径或默认路径
        model_path = model_path if model_path else default_model_path
        
        # 加载YOLOv8模型
        try:
            if os.path.exists(model_path):
                print(f"加载YOLO模型: {model_path}")
                self.model = YOLO(model_path)
            else:
                print(f"模型文件不存在: {model_path}, 尝试使用默认yolov8n模型")
                self.model = YOLO('yolov8n.pt')
            
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"加载YOLO模型失败: {str(e)}")
            raise
        
        # 初始化疲劳检测参数
        self.eyes_closed_threshold = 1.5  # 眼睛闭合超过1.5秒判定为疲劳
        self.yawn_threshold = 2.0  # 打哈欠超过2秒判定为疲劳
        self.head_down_threshold = 2.0  # 低头超过2秒判定为疲劳
        
        # 状态追踪变量
        self.closed_eyes_start_time = None
        self.yawn_start_time = None
        self.head_down_start_time = None
        
        # 疲劳警报计数
        self.fatigue_counter = 0
        self.last_alert_time = 0
        self.alert_cooldown = 5  # 警报冷却时间（秒）
        
        # 当前状态
        self.current_state = {
            'eyes_closed': False,
            'yawning': False,
            'head_down': False,
            'fatigue_detected': False,
            'fatigue_level': 0  # 0-100范围内的疲劳级别
        }
    
    def predict(self, frame):
        """
        对输入帧进行疲劳检测
        """
        try:
            # 使用YOLOv8模型进行检测
            results = self.model(frame, stream=True)
            
            # 获取当前时间
            current_time = time.time()
            
            # 重置当前检测到的特征
            eyes_detected = False
            mouth_open = False
            face_visible = False
            
            # 处理检测结果
            for result in results:
                # 获取检测框
                boxes = result.boxes
                
                # 分析检测到的目标
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 检测到人脸
                    if cls_id == 0 and conf > 0.5:
                        face_visible = True
                    
                    # 检测到睁开的眼睛
                    elif cls_id == 1 and conf > 0.5:
                        eyes_detected = True
                        if self.closed_eyes_start_time is not None:
                            self.closed_eyes_start_time = None
                            self.current_state['eyes_closed'] = False
                    
                    # 检测到闭合的眼睛
                    elif cls_id == 2 and conf > 0.5:
                        if self.closed_eyes_start_time is None:
                            self.closed_eyes_start_time = current_time
                        eyes_closed_duration = current_time - self.closed_eyes_start_time
                        self.current_state['eyes_closed'] = True
                        
                        if eyes_closed_duration >= self.eyes_closed_threshold:
                            self.trigger_fatigue_alert("检测到疲劳: 眼睛闭合过久", current_time)
                    
                    # 检测到打哈欠
                    elif cls_id == 3 and conf > 0.5:
                        mouth_open = True
                        if self.yawn_start_time is None:
                            self.yawn_start_time = current_time
                        yawn_duration = current_time - self.yawn_start_time
                        self.current_state['yawning'] = True
                        
                        if yawn_duration >= self.yawn_threshold:
                            self.trigger_fatigue_alert("检测到疲劳: 打哈欠", current_time)
                    
                    # 检测到低头
                    elif cls_id == 4 and conf > 0.5:
                        if self.head_down_start_time is None:
                            self.head_down_start_time = current_time
                        head_down_duration = current_time - self.head_down_start_time
                        self.current_state['head_down'] = True
                        
                        if head_down_duration >= self.head_down_threshold:
                            self.trigger_fatigue_alert("检测到疲劳: 长时间低头", current_time)
            
            # 如果没有检测到打哈欠，重置计时器
            if not mouth_open and self.yawn_start_time is not None:
                self.yawn_start_time = None
                self.current_state['yawning'] = False
            
            # 如果没有检测到低头，重置计时器
            if face_visible and self.head_down_start_time is not None:
                self.head_down_start_time = None
                self.current_state['head_down'] = False
            
            # 如果超过一段时间没有检测到疲劳，降低疲劳级别
            if current_time - self.last_alert_time > 10 and self.current_state['fatigue_level'] > 0:
                self.current_state['fatigue_level'] = max(0, self.current_state['fatigue_level'] - 5)
                if self.current_state['fatigue_level'] == 0:
                    self.current_state['fatigue_detected'] = False
            
            return self.current_state
            
        except Exception as e:
            print(f"疲劳检测出错: {str(e)}")
            return self.current_state
    
    def trigger_fatigue_alert(self, message, current_time):
        """触发疲劳警报"""
        # 检查冷却时间
        if current_time - self.last_alert_time > self.alert_cooldown:
            # 增加疲劳计数
            self.fatigue_counter += 1
            self.last_alert_time = current_time
            
            # 更新状态
            self.current_state['fatigue_detected'] = True
            
            # 增加疲劳级别，最高100
            self.current_state['fatigue_level'] = min(100, self.current_state['fatigue_level'] + 20)
    
    def detect_fatigue(self, frame):
        """
        检测视频帧中的疲劳迹象
        返回：带有标注的帧、是否检测到疲劳、当前状态
        """
        if frame is None or frame.size == 0:
            return frame, False, self.current_state
        
        # 进行疲劳检测
        state = self.predict(frame)
        
        # 在画面上显示疲劳级别
        self.draw_fatigue_level(frame, state['fatigue_level'])
        
        # 如果有疲劳警报，显示在画面上
        if state['fatigue_detected']:
            cv2.putText(frame, "疲劳警告!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, state['fatigue_detected'], state
    
    def draw_timer(self, frame, label, duration):
        """在画面上显示计时器"""
        text = f"{label}: {duration:.1f}秒"
        y_pos = 30 if label == "眼睛闭合" else (90 if label == "打哈欠" else 120)
        cv2.putText(frame, text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def draw_fatigue_level(self, frame, level):
        """在画面上显示疲劳级别"""
        # 定义文本和颜色
        text = f"疲劳级别: {level}%"
        
        # 根据疲劳级别选择颜色
        if level < 30:
            color = (0, 255, 0)  # 绿色
        elif level < 70:
            color = (0, 165, 255)  # 橙色
        else:
            color = (0, 0, 255)  # 红色
        
        # 绘制文本
        cv2.putText(frame, text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 绘制进度条
        bar_length = 200
        filled_length = int(bar_length * level / 100)
        
        # 进度条背景
        cv2.rectangle(frame, (10, 160), (10 + bar_length, 180), (200, 200, 200), -1)
        
        # 填充部分
        if filled_length > 0:
            cv2.rectangle(frame, (10, 160), (10 + filled_length, 180), color, -1)
        
        # 进度条边框
        cv2.rectangle(frame, (10, 160), (10 + bar_length, 180), (0, 0, 0), 1)

# 如果直接运行此文件，执行测试代码
if __name__ == "__main__":
    # 初始化疲劳检测器
    fatigue_detector = FatigueDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        # 读取一帧视频
        ret, frame = cap.read()
        if not ret:
            break
        
        # 进行疲劳检测
        annotated_frame, fatigue_detected, state = fatigue_detector.detect_fatigue(frame)
        
        # 显示处理后的帧
        cv2.imshow('疲劳检测', annotated_frame)
        
        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows() 