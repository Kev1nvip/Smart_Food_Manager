# app/backend/freshness_detector_simple.py
import numpy as np
import cv2
from typing import Dict
import random

class SimpleFreshnessDetector:
    """简化版新鲜度检测器（不依赖 PaddleClas）"""
    
    def __init__(self):
        self.labels = ["新鲜", "一般", "变质"]
    
    def predict(self, image_bytes: bytes) -> Dict:
        # 验证图片
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "error", "message": "图片解码失败"}
        
        # 基于图像特征的简单判断（示例）
        # 实际项目中可以用 OpenCV 分析颜色、纹理等
        brightness = np.mean(img)
        
        # 模拟检测逻辑
        if brightness > 150:  # 较亮
            scores = [0.8, 0.15, 0.05]  # 倾向新鲜
        elif brightness > 100:  # 中等
            scores = [0.3, 0.5, 0.2]   # 倾向一般
        else:  # 较暗
            scores = [0.1, 0.3, 0.6]   # 倾向变质
        
        # 加入随机性
        scores = [s + random.uniform(-0.1, 0.1) for s in scores]
        scores = [max(0, min(1, s)) for s in scores]
        total = sum(scores)
        scores = [s/total for s in scores]
        
        max_idx = scores.index(max(scores))
        
        return {
            "status": "success",
            "label": self.labels[max_idx],
            "score": float(scores[max_idx]),
            "all_results": [
                {"label": label, "score": float(score)}
                for label, score in zip(self.labels, scores)
            ]
        }