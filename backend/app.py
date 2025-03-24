"""
Flask应用主模块

提供人脸检测系统的Web服务
"""

from flask import Flask, render_template, request, Response, jsonify, session, redirect, url_for, send_file, g
import os
import cv2
import torch
import numpy as np
import sqlite3
import time
import json
import sys
import datetime
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import face_recognition
import shutil  # 用于文件操作
import uuid
import traceback

# 添加模型路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_dir = os.path.join(parent_dir, 'models', 'yolov7-face')
integration_dir = os.path.join(parent_dir, 'models')
sys.path.insert(0, model_dir)
sys.path.insert(0, integration_dir)
print(f"添加模型路径到系统路径: {model_dir}")
print(f"添加集成模块路径到系统路径: {integration_dir}")

# 直接导入YOLOv7-face相关模块
try:
    from models.experimental import attempt_load
    from utils.general import check_img_size, non_max_suppression, scale_coords
    from utils.torch_utils import select_device
    from face_db import init_db, get_all_faces, add_face_from_encoding, get_face_image_path, delete_face, delete_all_faces, reset_id_sequence
    print("成功导入YOLOv7-face模块")
except Exception as e:
    print(f"导入YOLOv7-face模块失败: {str(e)}")
    raise

# 定义路径常量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER_IMG_PATH = os.path.join(BASE_DIR, 'frontend', 'static', 'img', 'placeholder.jpg')
FACE_DB_PATH = os.path.join(BASE_DIR, 'models', 'yolov7-face', 'faces.db')
USER_DB_PATH = os.path.join(BASE_DIR, 'models', 'yolov7-face', 'users.db')

# 在应用启动前强制重建数据库
def migrate_face_db():
    """重建人脸数据库，处理 image_path 列缺失问题"""
    try:
        # 备份旧数据库
        if os.path.exists(FACE_DB_PATH):
            backup_path = FACE_DB_PATH + '.bak'
            shutil.copy2(FACE_DB_PATH, backup_path)
            print(f"已创建数据库备份: {backup_path}")
            
            # 从备份中读取编码数据
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT id, encoding FROM faces')
                old_faces = [(row[0], np.frombuffer(row[1], dtype=np.float64)) for row in cursor.fetchall()]
                conn.close()
                
                # 删除旧数据库并重新创建
                os.remove(FACE_DB_PATH)
                init_db(db_path=FACE_DB_PATH)
                
                # 重新插入旧的编码数据
                for face_id, encoding in old_faces:
                    add_face_from_encoding(encoding)
                
                print(f"数据库结构已更新，恢复了 {len(old_faces)} 条人脸数据")
                return True
            except Exception as e:
                conn.close()
                print(f"读取旧数据失败: {str(e)}")
                return False
        else:
            # 如果没有旧数据库，直接创建新的
            init_db(db_path=FACE_DB_PATH)
            print("创建了新的人脸数据库")
            return True
    except Exception as e:
        print(f"数据库迁移失败: {str(e)}")
        return False

# 初始化Flask应用
app = Flask(__name__, static_folder="../frontend/static", template_folder="../frontend/templates")
CORS(app)
app.secret_key = "humandetectionsystem2025"
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

# 初始化当前帧检测结果
app.current_frame_results = {}

# 初始化用户数据库
def init_user_db():
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        c = conn.cursor()
        
        # 创建用户表
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL)''')
        
        # 创建人脸表
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      image_path TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # 创建人员统计表
        c.execute('''CREATE TABLE IF NOT EXISTS person_stats
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      face_id INTEGER NOT NULL,
                      total_detections INTEGER DEFAULT 0,
                      last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      emotion_distribution TEXT,
                      average_fatigue REAL DEFAULT 0,
                      FOREIGN KEY (face_id) REFERENCES faces(id))''')
        
        # 创建检测记录表
        c.execute('''CREATE TABLE IF NOT EXISTS detection_records
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      face_id INTEGER,
                      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      faces_count INTEGER DEFAULT 0,
                      emotion TEXT,
                      fatigue_level REAL,
                      details TEXT,
                      FOREIGN KEY (face_id) REFERENCES faces(id))''')
        
        conn.commit()
        print("数据库表创建成功")
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
    finally:
        conn.close()

# 确保在应用启动时初始化数据库
init_user_db()

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 全局变量
camera = None  # 初始为None
detection_running = False
face_model = None
device = None
known_face_encodings = []
known_face_ids = []
frame_count = 0  # 添加帧计数器
detection_start_time = 0  # 添加检测开始时间变量
detection_stats = {
    "total_faces": 0,
    "timestamp": None,
    "last_10_records": [],
    "emotions": [],  # 存储情感检测结果
    "fatigue": {     # 存储疲劳检测结果
        "is_fatigued": False,
        "fatigue_level": 0,
        "state": {}
    }
}

# 添加情感和疲劳检测器
emotion_detector = None
fatigue_detector = None

# 添加全局检测设置变量
detection_settings = {
    "detection_threshold": 0.5,  # 降低检测阈值以提高召回率
    "face_padding": 5,  # 减少padding以减轻处理负担
    "similarity_threshold": 0.6,  # 提高相似度阈值使匹配更严格
    "vote_threshold": 2,  # 降低投票阈值
    "enable_preprocessing": False,  # 默认关闭预处理以提高性能
    "enable_voting": False,  # 默认关闭投票机制以提高性能
    "use_large_model": False,  # 默认使用小模型以提高性能
    "frame_skip": 2,  # 每隔2帧处理一次，即处理1帧，跳过2帧
    "enable_frame_skip": True,  # 是否启用帧跳过
    "enable_emotion_detection": False,  # 默认关闭情感检测
    "enable_fatigue_detection": False   # 默认关闭疲劳检测
}

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 添加全局变量用于存储人员统计数据
person_stats = {}
last_stats_update = 0
STATS_CLEANUP_INTERVAL = 3600  # 1小时清理一次过期数据

# 添加全局变量
last_detection_time = 0
detection_interval = 60  # 检测间隔（秒）
emotion_history = {}
fatigue_history = {}

# 路由：登录页面
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['username'] = username
            session['user_id'] = user[0]
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "用户名或密码不正确"})
    
    # GET请求时渲染login.html模板
    return render_template('login.html')

# 路由：注册
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"success": False, "message": "用户名和密码不能为空"})
    
    hashed_password = generate_password_hash(password)
    
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                      (username, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "用户名已存在"})

# 路由：注销
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

# 路由：检查登录状态
@app.route('/check_login')
def check_login():
    if 'username' in session:
        return jsonify({"logged_in": True, "username": session['username']})
    return jsonify({"logged_in": False})

# 加载YOLOv7人脸检测模型
def load_face_model():
    global face_model, device, emotion_detector, fatigue_detector
    if face_model is None:
        try:
            print("开始加载人脸检测模型...")
            
            # 使用用户指定的模型文件
            model_path = os.path.join('models', 'yolov7-face', 'weights', 'yolov7-face.pt')
            if not os.path.exists(model_path):
                model_path = os.path.join(BASE_DIR, 'models', 'yolov7-face', 'weights', 'yolov7-face.pt')
                if not os.path.exists(model_path):
                    # 尝试实验目录的模型
                    model_path = os.path.join(BASE_DIR, 'models', 'yolov7-face', 'runs', 'train', 'exp19', 'weights', 'best.pt')
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"找不到模型文件: {model_path}，当前目录: {os.getcwd()}")
            
            print(f"加载人脸模型文件: {model_path}")
            
            # 确保设备已正确配置
            if device is None:
                device = select_device('0' if torch.cuda.is_available() else 'cpu')
                print(f"选择设备: {device}")
            
            # 使用attempt_load函数从模型路径加载
            face_model = attempt_load(model_path, map_location=device)
            if face_model is None:
                raise ValueError("人脸模型加载失败，返回为None")
                
            face_model.to(device)
            face_model.eval()
            print(f"成功加载人脸检测模型，类型: {type(face_model)}")
            
            # 直接导入情感检测模型
            try:
                # 导入情感检测模块
                sys.path.append(os.path.join(BASE_DIR, 'models'))
                from emotion_detection.emotion_model import EmotionDetector
                
                # 加载情感检测模型
                emotion_model_path = os.path.join(BASE_DIR, 'models', 'emotion_detection', 'weights', 'emotion_vit_model.pth')
                if os.path.exists(emotion_model_path):
                    emotion_detector = EmotionDetector(model_path=emotion_model_path)
                    print(f"成功加载情感检测模型: {emotion_model_path}")
                else:
                    print(f"情感检测模型文件不存在: {emotion_model_path}")
                    # 尝试使用默认路径
                    default_model_path = os.path.join(BASE_DIR, 'models', 'emotion_detection', 'emotion_vit_model.pth')
                    if os.path.exists(default_model_path):
                        emotion_detector = EmotionDetector(model_path=default_model_path)
                        print(f"使用默认路径成功加载情感检测模型: {default_model_path}")
                    else:
                        print("无法找到情感检测模型文件，将使用随机初始化的模型")
                        emotion_detector = EmotionDetector()
            except Exception as e:
                print(f"加载情感检测模型失败: {str(e)}")
                emotion_detector = None
                
            # 直接导入疲劳检测模型
            try:
                # 导入疲劳检测模块
                from fatigue_detection.fatigue_model import FatigueDetector
                
                # 加载疲劳检测模型
                fatigue_model_path = os.path.join(BASE_DIR, 'models', 'fatigue_detection', 'weights', 'yolov8n.pt')
                if os.path.exists(fatigue_model_path):
                    fatigue_detector = FatigueDetector(model_path=fatigue_model_path)
                    print(f"成功加载疲劳检测模型: {fatigue_model_path}")
                else:
                    print(f"疲劳检测模型文件不存在: {fatigue_model_path}")
                    fatigue_detector = None
            except Exception as e:
                print(f"加载疲劳检测模型失败: {str(e)}")
                fatigue_detector = None
                
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            print(f"当前目录: {os.getcwd()}")
            print(f"目录内容: {os.listdir('.')}")
            if os.path.exists('models'):
                print(f"models目录内容: {os.listdir('models')}")
            raise

# 更新已知人脸列表
def update_known_faces():
    global known_face_encodings, known_face_ids
    faces = get_all_faces()
    known_face_encodings = []
    known_face_ids = []
    for face_id, encoding, _ in faces:
        known_face_encodings.append(encoding)
        known_face_ids.append(face_id)

# 人脸检测处理
def detect_faces(frame):
    global face_model, detection_settings
    if face_model is None:
        load_face_model()
    
    # 调整图像大小以提高性能
    img_size = 320  # 降低处理分辨率以提升性能
    
    # 准备输入
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR到RGB，HWC到CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        pred = face_model(img)[0]  # 获取第一个输出
    
    # 应用NMS
    conf_thres = detection_settings['detection_threshold']
    pred = non_max_suppression(pred, conf_thres, 0.5)
    
    # 处理检测结果
    faces = []
    if len(pred) > 0 and pred[0] is not None and len(pred[0]) > 0:
        # 将坐标从缩放的图像尺寸调整到原始图像尺寸
        det = pred[0].clone()
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
        
        for i in range(len(det)):
            bbox = det[i].cpu().numpy()
            
            # 检查数组长度，确保我们有足够的元素
            if len(bbox) >= 5:  # 至少需要4个坐标值和1个置信度
                # 提取前4个元素作为边界框坐标
                x1, y1, x2, y2 = bbox[:4]
                confidence = float(bbox[4])  # 置信度
                
                # 添加padding
                padding = detection_settings['face_padding']
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(frame.shape[1], int(x2) + padding)
                y2 = min(frame.shape[0], int(y2) + padding)
                
                # 提取人脸区域
                face_roi = frame[y1:y2, x1:x2]
                
                # 生成唯一ID（使用时间戳和随机数）
                face_id = f"face_{int(time.time()*1000)}_{i}"
                
                # 创建人脸信息字典
                face_info = {
                    'id': face_id,
                    'face_id': face_id,  # 添加face_id字段，与id保持一致
                    'bbox': [x1, y1, x2-x1, y2-y1],  # 改为[x, y, w, h]格式
                    'confidence': confidence,
                    'roi': face_roi
                }
                
                faces.append(face_info)
    
    return faces

def detect_emotion(face_roi):
    global emotion_detector
    if emotion_detector is None:
        try:
            from models.emotion_detection import EmotionDetector
            emotion_detector = EmotionDetector()
        except Exception as e:
            print(f"加载情绪检测模型失败: {str(e)}")
            return None, 0.0
    
    try:
        # 检查ROI是否为空或无效
        if face_roi is None or face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            print("情绪检测失败: 无效的人脸区域")
            return None, 0.0
            
        # 调整图像大小以适应模型
        face_roi = cv2.resize(face_roi, (224, 224))
        # 转换为RGB
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        # 归一化
        face_roi = face_roi.astype(np.float32) / 255.0
        # 添加批次维度
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # 进行预测
        try:
            emotion, confidence = emotion_detector.predict(face_roi)
            return emotion, confidence
        except ValueError as e:
            print(f"情感预测出错: {str(e)}")
            return None, 0.0
    except Exception as e:
        print(f"情绪检测失败: {str(e)}")
        return None, 0.0

def detect_fatigue(face_roi):
    global fatigue_detector
    if fatigue_detector is None:
        try:
            from models.fatigue_detection import FatigueDetector
            fatigue_detector = FatigueDetector()
        except Exception as e:
            print(f"加载疲劳检测模型失败: {str(e)}")
            return None, 0.0
    
    try:
        # 检查ROI是否为空或无效
        if face_roi is None or face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            print("疲劳检测失败: 无效的人脸区域")
            return None, 0.0
            
        # 调整图像大小以适应模型
        face_roi = cv2.resize(face_roi, (224, 224))
        # 转换为RGB
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        # 归一化
        face_roi = face_roi.astype(np.float32) / 255.0
        # 添加批次维度
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # 进行预测
        try:
            fatigue_level, confidence = fatigue_detector.predict(face_roi)
            return fatigue_level, confidence
        except Exception as e:
            print(f"疲劳预测出错: {str(e)}")
            return None, 0.0
    except Exception as e:
        print(f"疲劳检测失败: {str(e)}")
        return None, 0.0

# 处理检测结果并更新数据库
def process_detection_results(frame, faces):
    global detection_stats
    
    if not faces:
        return
        
    try:
        # 更新检测统计
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        faces_count = len(faces)
        
        # 初始化检测记录详情
        detection_details = {
            "total_detections": faces_count,
            "faces": []
        }
        
        # 处理每个检测到的人脸
        for face in faces:
            # 确保face包含必要的键
            if 'roi' not in face or face['roi'] is None or 'bbox' not in face:
                print(f"跳过不完整的人脸数据: {face.get('id', 'unknown')}")
                continue  # 跳过不完整的人脸数据
                
            face_id = face.get('face_id', None)
            if face_id is None:
                print("跳过没有face_id的人脸")
                continue  # 跳过没有face_id的人脸
                
            # 获取ROI
            face_roi = face['roi']
            
            # 确保ROI是有效的
            if face_roi is None or face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                print(f"跳过无效ROI的人脸: {face_id}")
                continue
                
            # 进行情感和疲劳检测
            try:
                emotion, emotion_conf = detect_emotion(face_roi)
            except Exception as e:
                print(f"情感检测异常: {str(e)}")
                emotion, emotion_conf = None, 0.0
                
            try:
                fatigue_level, fatigue_conf = detect_fatigue(face_roi)
            except Exception as e:
                print(f"疲劳检测异常: {str(e)}")
                fatigue_level, fatigue_conf = None, 0.0
            
            # 安全地更新face字典
            face['emotion'] = emotion if emotion is not None else "unknown"
            face['emotion_confidence'] = float(emotion_conf) if emotion_conf is not None else 0.0
            face['fatigue_level'] = float(fatigue_level) if fatigue_level is not None else 0.0
            face['fatigue_confidence'] = float(fatigue_conf) if fatigue_conf is not None else 0.0
            
            # 安全地添加到检测详情
            face_details = {
                "face_id": face_id,
                "bbox": face['bbox'],
                "confidence": float(face.get('confidence', 0.0)),
                "emotion": face['emotion'],
                "emotion_confidence": face['emotion_confidence'],
                "fatigue_level": face['fatigue_level'],
                "fatigue_confidence": face['fatigue_confidence']
            }
            detection_details["faces"].append(face_details)
        
        # 将检测结果保存到数据库
        conn = sqlite3.connect(USER_DB_PATH)
        c = conn.cursor()
        
        # 检查表是否存在
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detection_records'")
        if not c.fetchone():
            print("detection_records表不存在，正在初始化数据库...")
            init_user_db()
        
        # 插入检测记录
        details_json = json.dumps(detection_details)
        
        # 对每个检测到的人脸创建记录
        for face in faces:
            if 'face_id' not in face:
                continue
                
            face_id = face.get('face_id')
            emotion = face.get('emotion', "unknown")
            fatigue_level = face.get('fatigue_level', 0.0)
            
            # 插入检测记录
            try:
                c.execute('''
                    INSERT INTO detection_records 
                    (face_id, timestamp, faces_count, emotion, fatigue_level, details) 
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (face_id, timestamp, faces_count, emotion, fatigue_level, details_json))
            except sqlite3.Error as e:
                print(f"插入检测记录失败: {str(e)}")
                continue
            
            # 更新或创建人员统计记录
            try:
                c.execute("SELECT id FROM person_stats WHERE face_id = ?", (face_id,))
                person = c.fetchone()
                
                if person:
                    # 更新现有统计记录
                    # 先获取当前的情感分布
                    c.execute("SELECT emotion_distribution FROM person_stats WHERE face_id = ?", (face_id,))
                    emotion_dist_row = c.fetchone()
                    if emotion_dist_row and emotion_dist_row[0]:
                        try:
                            emotion_dist = json.loads(emotion_dist_row[0])
                        except json.JSONDecodeError:
                            emotion_dist = {}
                    else:
                        emotion_dist = {}
                    
                    # 更新情感分布
                    if emotion and emotion != "unknown":
                        emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
                    
                    # 安全处理fatigue_level
                    safe_fatigue = 0.0
                    if fatigue_level is not None:
                        try:
                            safe_fatigue = float(fatigue_level)
                        except (ValueError, TypeError):
                            safe_fatigue = 0.0
                    
                    # 更新总检测次数和最后检测时间
                    c.execute('''
                        UPDATE person_stats 
                        SET total_detections = total_detections + 1,
                            last_seen = ?,
                            emotion_distribution = ?,
                            average_fatigue = (average_fatigue * total_detections + ?) / (total_detections + 1)
                        WHERE face_id = ?
                    ''', (timestamp, json.dumps(emotion_dist), safe_fatigue, face_id))
                else:
                    # 创建新的统计记录
                    emotion_dist = {}
                    if emotion and emotion != "unknown":
                        emotion_dist[emotion] = 1
                    
                    # 安全处理fatigue_level
                    safe_fatigue = 0.0
                    if fatigue_level is not None:
                        try:
                            safe_fatigue = float(fatigue_level)
                        except (ValueError, TypeError):
                            safe_fatigue = 0.0
                    
                    c.execute('''
                        INSERT INTO person_stats 
                        (face_id, total_detections, last_seen, emotion_distribution, average_fatigue) 
                        VALUES (?, 1, ?, ?, ?)
                    ''', (face_id, timestamp, json.dumps(emotion_dist), safe_fatigue))
            except sqlite3.Error as e:
                print(f"更新人员统计记录失败: {str(e)}")
                continue
        
        # 提交所有数据库更改
        conn.commit()
        conn.close()
        
        # 更新全局统计信息
        detection_stats = {
            "current_faces": faces_count,
            "last_update": timestamp,
            "total_faces": detection_stats.get("total_faces", 0) + faces_count
        }
        
    except Exception as e:
        print(f"处理检测结果时出错: {str(e)}")
        traceback.print_exc()

# 视频流生成器
def generate_frames():
    global camera, face_model, emotion_detector, fatigue_detector, detection_running, frame_count, last_stats_update, person_stats
    
    while True:
        if not detection_running:
            # 如果检测未运行，返回一个静态帧
            static_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(static_frame, "摄像头未启动", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', static_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)
            continue
            
        if camera is None:
            # 尝试初始化摄像头
            for i in range(3):  # 尝试前3个摄像头设备
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    print(f"成功打开摄像头 {i}")
                    break
            if not camera.isOpened():
                print("无法打开任何摄像头")
                time.sleep(1)
                continue
        
        success, frame = camera.read()
        if not success:
            print("无法读取摄像头画面")
            time.sleep(1)
            continue
            
        try:
            # 人脸检测
            faces = detect_faces(frame)
            
            # 更新当前帧结果
            app.current_frame_results = {
                'faces': faces,
                'emotions': [],
                'fatigue': []
            }
            
            # 处理检测结果并更新统计数据
            process_detection_results(frame, faces)
            
            # 在帧上绘制人脸框和ID
            for face in faces:
                # 正确处理[x, y, width, height]格式的bbox
                x, y, w, h = face['bbox']
                face_id = face.get('id', 'Unknown')
                confidence = face.get('confidence', 0)
                
                # 绘制人脸框 - 从(x,y)到(x+w,y+h)
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                
                # 显示ID和置信度
                label = f"ID: {face_id} ({confidence:.2f})"
                cv2.putText(frame, label, (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示当前时间
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, current_time, (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示水印
            cv2.putText(frame, "Human Detection System", (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 编码帧
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 控制帧率
            time.sleep(0.03)  # 约30fps
            
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            time.sleep(0.1)  # 出错时稍微等待一下
            continue

# 路由：视频流
@app.route('/video_feed')
@login_required
def video_feed():
    print("请求视频流...")
    
    # 返回视频流，摄像头会在generate_frames函数中延迟初始化
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 路由：启动检测
@app.route('/start_detection', methods=['POST'])
@login_required
def start_detection():
    global detection_running, camera, detection_start_time
    if not detection_running:
        try:
            if camera is None:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    return jsonify({"success": False, "message": "无法打开摄像头"})
                
                # 设置摄像头参数
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                
                # 打印实际设置的参数
                width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = camera.get(cv2.CAP_PROP_FPS)
                print(f"摄像头实际设置: {width}x{height} @ {fps}fps")
            
            detection_running = True
            detection_start_time = time.time()  # 初始化检测开始时间
            return jsonify({"success": True})
        except Exception as e:
            print(f"启动检测失败: {str(e)}")
            return jsonify({"success": False, "message": str(e)})
    return jsonify({"success": True})

# 路由：停止检测
@app.route('/stop_detection')
@login_required
def stop_detection():
    global camera, detection_running
    try:
        detection_running = False
        if camera is not None:
            camera.release()
            camera = None
        return jsonify({'success': True, 'message': '摄像头已停止'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 路由：获取检测统计信息
@app.route('/detection_stats')
def get_detection_stats():
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # 检查person_stats表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_stats'")
        if not cursor.fetchone():
            print("获取检测统计信息失败: no such table: person_stats")
            # 初始化数据库
            init_user_db()
            return jsonify({
                'success': False,
                'error': 'person_stats表不存在，已尝试重新初始化'
            })
        
        # 获取最近24小时内的人脸统计
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT face_id) as unique_faces,
                COUNT(*) as total_detections,
                AVG(fatigue_level) as avg_fatigue
            FROM detection_records
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        
        stats = cursor.fetchone()
        if stats:
            unique_faces, total_detections, avg_fatigue = stats
        else:
            unique_faces, total_detections, avg_fatigue = 0, 0, 0
            
        # 获取情绪分布
        cursor.execute('''
            SELECT 
                emotion, 
                COUNT(*) as count
            FROM detection_records
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY emotion
        ''')
        
        emotions = {}
        for row in cursor.fetchall():
            if row[0]:  # 确保情绪标签不为空
                emotions[row[0]] = row[1]
        
        # 获取按小时统计的检测数
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count
            FROM detection_records
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY hour
            ORDER BY hour
        ''')
        
        hourly_data = {}
        for row in cursor.fetchall():
            hourly_data[row[0]] = row[1]
            
        # 填充缺失的小时
        for hour in range(24):
            hour_str = f"{hour:02d}"
            if hour_str not in hourly_data:
                hourly_data[hour_str] = 0
        
        # 按小时排序
        hourly_data = {k: hourly_data[k] for k in sorted(hourly_data.keys())}
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'unique_faces': unique_faces,
                'total_detections': total_detections,
                'average_fatigue': avg_fatigue if avg_fatigue else 0,
                'emotion_distribution': emotions,
                'hourly_detections': hourly_data,
                'current_faces': detection_stats.get('current_faces', 0),
                'last_update': detection_stats.get('last_update', None)
            }
        })
    except Exception as e:
        print(f"获取检测统计信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/face_records')
@login_required
def get_face_records():
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        c = conn.cursor()
        
        # 检查detection_records表是否存在
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detection_records'")
        if not c.fetchone():
            print("获取检测记录失败: no such table: detection_records")
            # 初始化数据库
            init_user_db()
            return jsonify({
                'success': False,
                'error': 'detection_records表不存在，已尝试重新初始化'
            })
        
        # 获取最近100条检测记录
        c.execute('''
            SELECT id, face_id, timestamp, faces_count, emotion, fatigue_level, details
            FROM detection_records
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        
        records = []
        for row in c.fetchall():
            rec_id, face_id, timestamp, faces_count, emotion, fatigue_level, details = row
            
            # 解析details JSON
            details_dict = json.loads(details) if details else {}
            
            records.append({
                'id': rec_id,
                'face_id': face_id,
                'timestamp': timestamp,
                'faces_count': faces_count,
                'emotion': emotion,
                'fatigue_level': fatigue_level,
                'details': details_dict
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'records': records
        })
    except Exception as e:
        print(f"获取检测记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/person_stats')
@login_required
def get_person_stats():
    try:
        # 检查人员统计数据
        stats = get_all_face_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        print(f"获取人员统计数据失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def get_all_face_stats():
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    
    # 检查必要的表是否存在
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_stats'")
    if not c.fetchone():
        print("获取人员统计数据失败: no such table: person_stats")
        # 初始化数据库
        init_user_db()
        return []
    
    try:
        # 获取所有人员统计信息
        c.execute('''
            SELECT 
                ps.face_id,
                ps.total_detections,
                ps.last_seen,
                ps.emotion_distribution,
                ps.average_fatigue,
                f.image_path
            FROM person_stats ps
            LEFT JOIN faces f ON ps.face_id = f.id
            ORDER BY ps.last_seen DESC
        ''')
        
        stats = []
        for row in c.fetchall():
            face_id, total_detections, last_seen, emotion_dist, avg_fatigue, image_path = row
            
            # 获取情绪历史
            try:
                c.execute('''
                    SELECT emotion, COUNT(*) as count
                    FROM detection_records
                    WHERE face_id = ?
                    GROUP BY emotion
                ''', (face_id,))
                emotion_history = [{'emotion': e[0], 'count': e[1]} for e in c.fetchall() if e[0]]
            except sqlite3.OperationalError:
                emotion_history = []
            
            # 获取疲劳历史
            try:
                c.execute('''
                    SELECT fatigue_level, timestamp
                    FROM detection_records
                    WHERE face_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (face_id,))
                fatigue_history = [{'level': f[0], 'timestamp': f[1]} for f in c.fetchall() if f[0] is not None]
            except sqlite3.OperationalError:
                fatigue_history = []
            
            stats.append({
                'face_id': face_id,
                'total_detections': total_detections,
                'last_seen': last_seen,
                'emotion_distribution': json.loads(emotion_dist) if emotion_dist else {},
                'average_fatigue': avg_fatigue,
                'image_path': image_path,
                'emotion_history': emotion_history,
                'fatigue_history': fatigue_history
            })
    except Exception as e:
        print(f"获取人员统计数据时出错: {str(e)}")
        conn.close()
        return []
    
    conn.close()
    return stats

# 路由：获取人脸数据库中的所有人脸
@app.route('/faces')
@login_required
def get_faces():
    # 获取所有已知人脸ID及其图像路径
    face_data = []
    try:
        # 尝试获取包含image_path的新数据结构
        faces = get_all_faces()
        for face_id, encoding, image_path in faces:
            face_info = {"id": face_id}
            if image_path:
                face_info["image_path"] = image_path
            face_data.append(face_info)
    except Exception as e:
        # 如果出错（例如数据库结构旧），尝试只获取ID
        try:
            print(f"获取带图像路径的人脸数据失败: {str(e)}，尝试仅获取ID")
            conn = sqlite3.connect(FACE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM faces')
            for row in cursor.fetchall():
                face_data.append({"id": row[0]})
            conn.close()
        except Exception as e2:
            print(f"获取人脸ID失败: {str(e2)}")
    
    return jsonify(face_data)

# 新增API：获取人脸图像
@app.route('/face_image/<int:face_id>')
@login_required
def face_image(face_id):
    try:
        image_path = get_face_image_path(face_id)
        if image_path and os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"获取人脸图像错误: {str(e)}")
    
    # 如果没有找到图像或出错，返回默认图像
    return send_file(PLACEHOLDER_IMG_PATH, mimetype='image/jpeg')

# 修改删除人脸API
@app.route('/delete_face/<int:face_id>', methods=['DELETE'])
@login_required
def delete_face_route(face_id):
    try:
        delete_face(face_id)
        return jsonify({"success": True, "message": f"已删除ID为{face_id}的人脸记录"})
    except Exception as e:
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"})

# 路由：清空人脸数据库
@app.route('/clear_faces', methods=['POST'])
@login_required
def clear_faces():
    try:
        delete_all_faces(db_path=FACE_DB_PATH)
        # 重新初始化全局变量
        global known_face_encodings, known_face_ids
        known_face_encodings = []
        known_face_ids = []
        return jsonify({"success": True, "message": "已清空人脸数据库"})
    except Exception as e:
        return jsonify({"success": False, "message": f"清空失败: {str(e)}"})

# 路由：重置人脸ID序列
@app.route('/reset_face_sequence', methods=['POST'])
@login_required
def reset_face_sequence():
    """重置人脸ID序列，确保ID连续"""
    try:
        # 获取所有记录并按ID排序
        conn = sqlite3.connect('models/yolov7-face/faces.db')
        cursor = conn.cursor()
        
        # 获取所有记录
        cursor.execute('SELECT id, encoding, image_path FROM faces ORDER BY id')
        records = cursor.fetchall()
        
        # 删除所有记录
        cursor.execute('DELETE FROM faces')
        
        # 重新插入记录，使用新的连续ID
        for new_id, (old_id, encoding, image_path) in enumerate(records, 1):
            # 如果图像路径存在，更新路径中的ID
            if image_path:
                new_path = image_path.replace(f'face_{old_id}.jpg', f'face_{new_id}.jpg')
                # 重命名文件
                if os.path.exists(image_path):
                    os.rename(image_path, new_path)
            else:
                new_path = None
            
            # 插入新记录
            cursor.execute(
                'INSERT INTO faces (id, encoding, image_path) VALUES (?, ?, ?)',
                (new_id, encoding, new_path)
            )
        
        # 重置自增序列
        cursor.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'faces'", (len(records),))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'成功重置{len(records)}条人脸记录'
        })
    except Exception as e:
        print(f"重置人脸序列失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'重置人脸序列失败: {str(e)}'
        })

# 路由：控制面板
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# 主页重定向到前端
@app.route('/')
def index():
    return redirect('/login')

# 路由：更新检测设置
@app.route('/update_detection_settings', methods=['POST'])
@login_required
def update_detection_settings():
    global detection_settings
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "未提供数据"})
        
        # 更新设置
        for key, value in data.items():
            if key in detection_settings:
                # 对于特定类型的值进行类型转换
                if key in ['detection_threshold', 'similarity_threshold'] and value:
                    detection_settings[key] = float(value)
                elif key in ['face_padding', 'vote_threshold', 'frame_skip'] and value:
                    detection_settings[key] = int(value)
                elif key in ['enable_preprocessing', 'enable_voting', 'use_large_model', 
                            'enable_frame_skip', 'enable_emotion_detection', 'enable_fatigue_detection']:
                    detection_settings[key] = bool(value)
                else:
                    detection_settings[key] = value
        
        # 检查并初始化情感检测模型
        if detection_settings.get('enable_emotion_detection', False):
            try:
                from models.emotion_detection import EmotionDetector
                print("情感检测模型已启用")
            except ImportError:
                print("无法导入情感检测模型")
                detection_settings['enable_emotion_detection'] = False
        
        # 检查并初始化疲劳检测模型
        if detection_settings.get('enable_fatigue_detection', False):
            try:
                from models.fatigue_detection import FatigueDetector
                print("疲劳检测模型已启用")
            except ImportError:
                print("无法导入疲劳检测模型")
                detection_settings['enable_fatigue_detection'] = False
        
        print(f"检测设置已更新: {detection_settings}")
        return jsonify({"success": True, "settings": detection_settings})
    except Exception as e:
        print(f"更新检测设置出错: {str(e)}")
        return jsonify({"success": False, "message": f"更新设置出错: {str(e)}"})

# 获取当前检测设置的API
@app.route('/detection_settings')
@login_required
def get_detection_settings():
    # 返回当前检测设置
    return jsonify(detection_settings)

# 删除单个检测记录
@app.route('/delete_record/<int:record_id>', methods=['DELETE'])
@login_required
def delete_record(record_id):
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # 检查记录是否存在
        cursor.execute("SELECT id FROM detection_records WHERE id = ?", (record_id,))
        record = cursor.fetchone()
        
        if not record:
            conn.close()
            return jsonify({"success": False, "message": f"未找到ID为{record_id}的记录"})
        
        # 删除记录
        cursor.execute("DELETE FROM detection_records WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": f"已删除ID为{record_id}的记录"})
    except Exception as e:
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"})

@app.route('/recent_detections')
def get_recent_detections():
    """获取最近的检测记录"""
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # 检查必要的表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detection_records'")
        if not cursor.fetchone():
            print("获取最近检测记录失败: no such table: detection_records")
            # 初始化数据库
            init_user_db()
            return jsonify({
                'success': False,
                'error': 'detection_records表不存在，已尝试重新初始化'
            })
        
        # 获取最近10条检测记录
        cursor.execute('''
            SELECT timestamp, faces_count, details
            FROM detection_records
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        
        records = []
        for row in cursor.fetchall():
            details = json.loads(row[2]) if row[2] else {}
            records.append({
                'timestamp': row[0],
                'faces_count': row[1],
                'details': details
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'records': records
        })
    except Exception as e:
        print(f"获取最近检测记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)