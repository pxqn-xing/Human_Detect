#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人脸数据库管理模块
支持人脸编码和图像存储
"""

import sqlite3
import os
import json
from datetime import datetime
import numpy as np
import cv2
import face_recognition
import time

# 数据库和图像保存的路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(CURRENT_DIR, 'faces.db')
FACES_DIR = os.path.join(CURRENT_DIR, 'face_images')

# 确保人脸图像目录存在
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
    print(f"创建人脸图像目录: {FACES_DIR}")

def init_db(db_path=DEFAULT_DB_PATH):
    """初始化数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建检测记录表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detection_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        faces_count INTEGER NOT NULL,
        details TEXT NOT NULL,
        face_ids TEXT,
        frame_path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 创建人员统计表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS person_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL,
        emotion TEXT,
        emotion_confidence REAL,
        fatigue_level REAL,
        fatigue_confidence REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (face_id) REFERENCES faces(id)
    )
    ''')
    
    # 创建人脸表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_encoding BLOB NOT NULL,
        face_image_path TEXT,
        total_detections INTEGER DEFAULT 0,
        last_seen DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"数据库初始化完成: {db_path}")

def add_face_from_encoding(encoding: np.ndarray, face_image=None, db_path=None):
    """添加人脸编码到数据库，可选保存人脸图像"""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    # 转换numpy数组为blob
    encoding_blob = sqlite3.Binary(encoding.tobytes())
    
    # 如果提供了人脸图像，保存它
    image_path = None
    if face_image is not None:
        cursor.execute('SELECT MAX(id) FROM faces')
        result = cursor.fetchone()
        next_id = 1 if result[0] is None else result[0] + 1
        
        image_filename = f'face_{next_id}.jpg'
        image_path = os.path.join(FACES_DIR, image_filename)
        
        cv2.imwrite(image_path, face_image)
        print(f"保存人脸图像: {image_path}")
        
        # 插入记录包含图像路径和初始统计信息
        cursor.execute('''
        INSERT INTO faces (face_encoding, face_image_path, total_detections, last_seen, emotion_history, fatigue_history, created_at)
        VALUES (?, ?, 0, CURRENT_TIMESTAMP, '[]', '[]', CURRENT_TIMESTAMP)
        ''', (encoding_blob, image_path))
    else:
        cursor.execute('''
        INSERT INTO faces (face_encoding, total_detections, last_seen, emotion_history, fatigue_history, created_at)
        VALUES (?, 0, CURRENT_TIMESTAMP, '[]', '[]', CURRENT_TIMESTAMP)
        ''', (encoding_blob,))
    
    conn.commit()
    
    cursor.execute('SELECT last_insert_rowid()')
    new_id = cursor.fetchone()[0]
    
    conn.close()
    return new_id

def update_face_stats(face_id, emotion=None, emotion_conf=None, fatigue_level=None, fatigue_conf=None):
    """更新人脸统计信息"""
    try:
        conn = sqlite3.connect('models/yolov7-face/faces.db')
        cursor = conn.cursor()
        
        # 更新人脸统计信息
        cursor.execute('''
            INSERT INTO person_stats (
                face_id, timestamp, emotion, emotion_confidence, 
                fatigue_level, fatigue_confidence
            ) VALUES (?, datetime('now'), ?, ?, ?, ?)
        ''', (face_id, emotion, emotion_conf, fatigue_level, fatigue_conf))
        
        # 更新人脸的最后检测时间
        cursor.execute('''
            UPDATE faces 
            SET last_seen = datetime('now'),
                total_detections = total_detections + 1
            WHERE id = ?
        ''', (face_id,))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"更新人脸统计信息失败: {str(e)}")
        if conn:
            conn.close()

def get_face_stats(face_id: int, db_path=None):
    """获取人脸统计信息"""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT total_detections, last_seen, emotion_history, fatigue_history
    FROM faces WHERE id = ?
    ''', (face_id,))
    
    result = cursor.fetchone()
    if result:
        stats = {
            'total_detections': result[0],
            'last_seen': result[1],
            'emotion_history': eval(result[2]),
            'fatigue_history': eval(result[3])
        }
    else:
        stats = None
    
    conn.close()
    return stats

def get_all_face_stats():
    """获取所有人脸的统计信息"""
    try:
        conn = sqlite3.connect('models/yolov7-face/faces.db')
        cursor = conn.cursor()
        
        # 获取所有人脸的统计信息
        cursor.execute('''
            SELECT 
                f.id as face_id,
                f.total_detections,
                f.last_seen,
                GROUP_CONCAT(p.emotion) as emotions,
                GROUP_CONCAT(p.emotion_confidence) as emotion_confidences,
                GROUP_CONCAT(p.fatigue_level) as fatigue_levels,
                GROUP_CONCAT(p.fatigue_confidence) as fatigue_confidences
            FROM faces f
            LEFT JOIN person_stats p ON f.id = p.face_id
            WHERE p.timestamp >= datetime('now', '-24 hours')
            GROUP BY f.id
        ''')
        
        stats = []
        for row in cursor.fetchall():
            face_id, total_detections, last_seen, emotions, emotion_confidences, fatigue_levels, fatigue_confidences = row
            
            # 处理情绪历史
            emotion_history = []
            if emotions and emotion_confidences:
                emotions_list = emotions.split(',')
                confidences_list = emotion_confidences.split(',')
                for emotion, conf in zip(emotions_list, confidences_list):
                    if emotion and conf:
                        emotion_history.append({
                            'emotion': emotion,
                            'confidence': float(conf)
                        })
            
            # 处理疲劳历史
            fatigue_history = []
            if fatigue_levels and fatigue_confidences:
                levels_list = fatigue_levels.split(',')
                confidences_list = fatigue_confidences.split(',')
                for level, conf in zip(levels_list, confidences_list):
                    if level and conf:
                        fatigue_history.append({
                            'fatigue_level': float(level),
                            'confidence': float(conf)
                        })
            
            stats.append({
                'face_id': face_id,
                'total_detections': total_detections,
                'last_seen': last_seen,
                'emotion_history': emotion_history,
                'fatigue_history': fatigue_history
            })
        
        conn.close()
        return stats
        
    except Exception as e:
        print(f"获取人脸统计信息失败: {str(e)}")
        if conn:
            conn.close()
        return []

def get_all_faces(db_path=None):
    """获取所有人脸编码和图像路径"""
    # 为线程安全创建新连接
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    # 获取所有记录
    cursor.execute('SELECT id, face_encoding, face_image_path FROM faces')
    rows = cursor.fetchall()
    
    # 处理结果
    result = []
    for row in rows:
        face_id = row[0]
        encoding = np.frombuffer(row[1], dtype=np.float64)
        image_path = row[2]  # 可能为None
        result.append((face_id, encoding, image_path))
    
    conn.close()
    return result

def get_face_image_path(face_id, db_path=None):
    """获取特定ID的人脸图像路径"""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT face_image_path FROM faces WHERE id = ?', (face_id,))
    result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else None

def reset_id_sequence(db_path=None):
    """重置SQLite自增ID序列"""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    # 检查表是否为空
    cursor.execute("SELECT COUNT(*) FROM faces")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # 如果表为空，重置序列
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='faces'")
        conn.commit()
        print("重置人脸ID序列，下一个ID将从1开始")
    else:
        # 获取当前最大ID
        cursor.execute("SELECT MAX(id) FROM faces")
        max_id = cursor.fetchone()[0]
        
        # 更新序列，确保下一个ID是最大ID+1
        cursor.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'faces'", (max_id,))
        conn.commit()
        print(f"更新人脸ID序列，下一个ID将从{max_id+1}开始")
    
    conn.close()
    return True

def delete_face(face_id, db_path=None):
    """删除指定ID的人脸及其图像"""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    
    # 获取图像路径
    cursor.execute('SELECT face_image_path FROM faces WHERE id = ?', (face_id,))
    result = cursor.fetchone()
    
    # 删除数据库记录
    cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
    conn.commit()
    
    # 如果有关联图像，也删除文件
    if result and result[0] and os.path.exists(result[0]):
        try:
            os.remove(result[0])
            print(f"删除人脸图像: {result[0]}")
        except Exception as e:
            print(f"删除图像文件失败: {str(e)}")
    
    conn.close()
    
    # 检查是否需要重置ID序列
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM faces")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        # 如果表为空，重置ID序列
        reset_id_sequence(db_path)
    
    return True

def delete_all_faces(db_path=None):
    """删除所有人脸记录和图像"""
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有图像路径
    cursor.execute('SELECT id, face_image_path FROM faces')
    cursor.execute('SELECT id, image_path FROM faces')
    rows = cursor.fetchall()
    
    # 删除所有数据库记录
    cursor.execute('DELETE FROM faces')
    conn.commit()
    
    # 重置ID序列
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='faces'")
    conn.commit()
    conn.close()
    
    # 删除所有关联图像文件
    deleted_count = 0
    for _, image_path in rows:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                deleted_count += 1
            except Exception as e:
                print(f"删除图像文件失败: {str(e)}")
    
    print(f"已删除所有{len(rows)}条人脸记录和{deleted_count}个图像，ID序列已重置")
    return True

def cleanup_expired_stats():
    """清理过期的统计数据"""
    try:
        conn = sqlite3.connect('models/yolov7-face/faces.db')
        cursor = conn.cursor()
        
        # 删除24小时前的统计记录
        cursor.execute('''
        DELETE FROM person_stats 
        WHERE timestamp < datetime('now', '-24 hours')
        ''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"清理过期数据失败: {str(e)}")
        if conn:
            conn.close()

if __name__ == "__main__":
    # 初始化数据库
    init_db()
    print("人脸数据库已初始化")
