# test_freshness.py
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.backend.freshness_detector import FreshnessDetector
'''
def test_freshness_detection():
    detector = FreshnessDetector(device="cpu")
    
    with open("D:\Smart_Food_Manager\\tests\images\\fresh_apple.jpg", "rb") as f:
        img_bytes = f.read()
    
    result = detector.predict(img_bytes)
    
    assert result["status"] == "success"
    assert result["label"] in ["新鲜", "一般", "变质"]
    assert 0 <= result["score"] <= 1
    assert len(result["all_results"]) == 3
    
if __name__ == "__main__":
    test_freshness_detection()
'''

# 初始化检测器
detector = FreshnessDetector()

# 预测新鲜度
with open("D:\Smart_Food_Manager\\tests\images\spoiled_banana.jpg", "rb") as f:
    image_bytes = f.read()
    
result = detector.predict(image_bytes)
# 打印检测结果
if result["status"] == "success":
    print(f"检测结果: {result['label']} (置信度: {result['score']:.2f})")
    advice = detector.get_freshness_advice(result["label"], result["score"])
    print(f"建议: {advice}")
else:
    print(f"检测失败: {result['message']}")
    

