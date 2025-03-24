import cv2
import torch
import torchvision.transforms as transforms
import timm
import numpy as np
import os

# 定义情感类别（中英文对照）
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_labels_zh = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '中性']

class EmotionDetector:
    def __init__(self, model_path=None):
        """
        初始化情感检测模型
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"情感检测模型使用设备: {self.device}")
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建模型，不使用预训练权重
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
        
        # 修改最后一层全连接层以适应7种情感分类
        num_ftrs = self.model.head.in_features
        self.model.head = torch.nn.Linear(num_ftrs, len(emotion_labels))
        
        # 设置默认模型路径
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'weights', 'emotion_vit_model.pth')
        
        # 加载模型权重
        if os.path.exists(model_path):
            print(f"加载情感检测模型权重: {model_path}")
            try:
                # 首先尝试加载完整模型
                state_dict = torch.load(model_path, map_location=self.device)
                if 'head.weight' in state_dict and state_dict['head.weight'].shape[0] == len(emotion_labels):
                    self.model.load_state_dict(state_dict)
                    print("成功加载完整情感模型权重")
                else:
                    print("模型权重维度不匹配，尝试加载主干网络权重...")
                    # 过滤掉不匹配的层
                    filtered_dict = {k: v for k, v in state_dict.items() 
                                   if k in self.model.state_dict() and 'head' not in k}
                    self.model.load_state_dict(filtered_dict, strict=False)
                    print("成功加载主干网络权重，分类头将使用随机初始化")
            except Exception as e:
                print(f"加载模型失败: {str(e)}")
                print("使用随机初始化的模型")
        else:
            print(f"警告: 找不到模型权重文件 {model_path}")
            print("使用随机初始化的模型")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("警告：无法加载人脸级联分类器，请确保OpenCV正确安装")
    
    def predict(self, face_roi):
        """
        对输入的人脸ROI进行情感预测
        """
        try:
            # 图像预处理
            input_tensor = self.transform(face_roi)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # 情感预测
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                emotion_idx = predicted.item()
                emotion_en = emotion_labels[emotion_idx]
                emotion_zh = emotion_labels_zh[emotion_idx]
                conf_value = confidence.item()
            
            return {
                'emotion_en': emotion_en,
                'emotion_zh': emotion_zh,
                'confidence': conf_value
            }
        except Exception as e:
            print(f"情感预测出错: {str(e)}")
            return None
    
    def detect_emotion(self, frame):
        """
        对输入的图像帧进行情感检测
        返回添加了情感标签的图像帧和检测到的情感数据
        """
        if frame is None or frame.size == 0:
            return frame, []
        
        # 转换为RGB (如果是BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        
        # 存储检测结果
        emotion_results = []
        
        for (x, y, w, h) in faces:
            try:
                # 提取人脸ROI
                face_roi = rgb_frame[y:y+h, x:x+w]
                
                # 进行情感预测
                emotion_result = self.predict(face_roi)
                if emotion_result:
                    # 在图像上绘制人脸框和情感标签
                    color = (0, 255, 0)  # 绿色
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 显示中文和英文情感标签以及置信度
                    label = f"{emotion_result['emotion_zh']}/{emotion_result['emotion_en']}: {emotion_result['confidence']:.2f}"
                    cv2.putText(frame, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # 添加到结果列表
                    emotion_results.append({
                        'bbox': (x, y, w, h),
                        **emotion_result
                    })
            
            except Exception as e:
                print(f"情感检测出错: {str(e)}")
                # 在出错时仍然画出人脸框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        return frame, emotion_results
    
    def analyze_emotion_distribution(self, emotions_history, window_size=30):
        """
        分析一段时间内的情感分布
        emotions_history: 情感历史记录列表
        window_size: 分析窗口大小（帧数）
        返回最近window_size帧的情感分布统计
        """
        if not emotions_history:
            return {}
        
        # 取最近的window_size帧
        recent_emotions = emotions_history[-window_size:] if len(emotions_history) > window_size else emotions_history
        
        # 统计各情感出现次数
        emotion_counts = {emotion: 0 for emotion in emotion_labels}
        for frame_emotions in recent_emotions:
            for emotion_data in frame_emotions:
                emotion_counts[emotion_data['emotion_en']] += 1
        
        # 计算百分比
        total = sum(emotion_counts.values())
        if total > 0:
            emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
        else:
            emotion_percentages = {emotion: 0 for emotion in emotion_labels}
        
        return emotion_percentages
    
    def get_dominant_emotion(self, emotions_history, window_size=30):
        """
        获取一段时间内的主导情感
        emotions_history: 情感历史记录列表
        window_size: 分析窗口大小（帧数）
        返回出现频率最高的情感及其百分比
        """
        distribution = self.analyze_emotion_distribution(emotions_history, window_size)
        if not distribution:
            return None, 0
            
        dominant_emotion = max(distribution, key=distribution.get)
        percentage = distribution[dominant_emotion]
        
        return dominant_emotion, percentage

# 测试代码
if __name__ == "__main__":
    # 初始化情感检测器
    emotion_detector = EmotionDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 情感历史记录
    emotions_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测情感
        frame, emotions = emotion_detector.detect_emotion(frame)
        
        # 添加到历史记录
        emotions_history.append(emotions)
        
        # 如果历史记录过长，裁剪它
        if len(emotions_history) > 100:
            emotions_history = emotions_history[-100:]
        
        # 获取主导情感
        if len(emotions_history) > 10:
            dominant_emotion, percentage = emotion_detector.get_dominant_emotion(emotions_history)
            if dominant_emotion:
                # 在帧上显示主导情感
                text = f"主导情感: {dominant_emotion} ({percentage:.1f}%)"
                cv2.putText(frame, text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow('情感检测', frame)
        
        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows() 