# test_freshness_detector.py
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.backend.freshness_detector_simple import SimpleFreshnessDetector
import cv2
import numpy as np

def test_freshness_detector():
    # 1. 初始化检测器
    detector = SimpleFreshnessDetector()
    print("检测器初始化成功")
    
    # 2. 创建测试图片
    # 创建亮色图片（模拟新鲜食品）
    bright_img = np.full((300, 300, 3), 200, dtype=np.uint8)  # 亮色
    cv2.putText(bright_img, "Fresh", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # 创建暗色图片（模拟变质食品）
    dark_img = np.full((300, 300, 3), 50, dtype=np.uint8)  # 暗色
    cv2.putText(dark_img, "Spoiled", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # 3. 测试亮色图片（预期：新鲜）
    _, bright_bytes = cv2.imencode('.jpg', bright_img)
    bright_result = detector.predict(bright_bytes.tobytes())
    print("\n亮色图片检测结果:")
    print_result(bright_result)
    
    # 4. 测试暗色图片（预期：变质）
    _, dark_bytes = cv2.imencode('.jpg', dark_img)
    dark_result = detector.predict(dark_bytes.tobytes())
    print("\n暗色图片检测结果:")
    print_result(dark_result)
    
    # 5. 测试无效图片
    invalid_result = detector.predict(b"invalid_image_data")
    print("\n无效图片检测结果:")
    print_result(invalid_result)

def print_result(result):
    """格式化打印检测结果"""
    if result["status"] == "success":
        print(f"状态: {result['status']}")
        print(f"检测结果: {result['label']} (置信度: {result['score']:.2f})")
        print("详细结果:")
        for item in result["all_results"]:
            print(f"  - {item['label']}: {item['score']:.2f}")
    else:
        print(f"状态: {result['status']}")
        print(f"错误信息: {result['message']}")

def test_with_real_images():
    detector = SimpleFreshnessDetector()
    
    # 新鲜水果图片
    with open("tests/images/fresh_apple.jpg", "rb") as f:
        fresh_result = detector.predict(f.read())
        print("新鲜苹果检测结果:", fresh_result["label"])
        print_result(fresh_result)
    
    # 变质食品图片
    with open("tests/images/spoiled_banana.jpg", "rb") as f:
        spoiled_result = detector.predict(f.read())
        print("变质香蕉检测结果:", spoiled_result["label"])
        print_result(spoiled_result)
        
if __name__ == "__main__":
    #test_freshness_detector()
    test_with_real_images()