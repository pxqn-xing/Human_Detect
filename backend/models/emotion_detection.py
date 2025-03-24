import numpy as np
import tensorflow as tf

class EmotionDetector:
    def __init__(self):
        # 这里应该加载预训练模型，但为了演示，我们使用一个简单的模型
        self.model = self._create_model()
        
    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return model
    
    def predict(self, face_roi):
        # 这里应该使用真实的模型进行预测，但为了演示，我们返回随机结果
        emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
        emotion = np.random.choice(emotions)
        confidence = np.random.uniform(0.5, 1.0)
        return emotion, confidence 