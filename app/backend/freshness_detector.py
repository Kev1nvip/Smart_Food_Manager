from paddle.vision.models import mobilenet_v3_large
import paddle
import numpy as np
import cv2
import logging
import os
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class FreshnessDetector:
    """食品新鲜度检测器"""
    
    def __init__(self, 
                 model_name: str = "MobileNetV3_large",
                 model_path: Optional[str] = None,
                 label_list: Optional[list] = None,
                 device: str = "cpu"):
        """
        初始化新鲜度检测器
        
        Args:
            model_name: 模型名称（当前仅支持 "MobileNetV3_large"）
            model_path: 自定义模型权重路径 (.pdparams)
            label_list: 类别标签列表，如 ["新鲜", "轻微变质", "严重变质"]
            device: 使用设备 "gpu" 或 "cpu"
        """
        self.label_list = label_list or ["新鲜", "一般", "变质"]
        self.device = device
        self.model = None
        
        try:
            # 设置设备
            paddle.set_device(device)
            
            # 初始化模型
            self.model = mobilenet_v3_large(pretrained=False, num_classes=len(self.label_list))
            
            # 加载自定义权重（如果有）
            if model_path and os.path.exists(model_path):
                logger.info(f"加载自定义模型权重: {model_path}")
                state_dict = paddle.load(model_path)
                self.model.set_state_dict(state_dict)
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info(f"新鲜度检测器初始化成功 ({device.upper()})")
        except Exception as e:
            logger.error(f"新鲜度检测器初始化失败: {e}", exc_info=True)
            raise RuntimeError(f"检测器初始化失败: {str(e)}")
    
    def predict(self, image_bytes: bytes) -> Dict[str, any]:
        """
        预测食品新鲜度
        
        Args:
            image_bytes: 图片二进制数据
            
        Returns:
            {
                "status": "success/error",
                "label": "新鲜",
                "score": 0.95,
                "all_results": [
                    {"label": "新鲜", "score": 0.95},
                    {"label": "一般", "score": 0.03},
                    {"label": "变质", "score": 0.02}
                ],
                "message": "错误信息（如果有）"
            }
        """
        if self.model is None:
            return {
                "status": "error",
                "message": "检测器未初始化"
            }
        
        try:
            # 解码图片
            img = self._decode_image(image_bytes)
            if img is None:
                return {
                    "status": "error",
                    "message": "图片解码失败"
                }
            
            # 预处理
            img = self._preprocess_image(img)
            
            # 转换为张量 - 确保数据类型正确
            img_tensor = paddle.to_tensor(img, dtype='float32').unsqueeze(0)  # 添加批次维度
            
            # 执行预测
            with paddle.no_grad():
                output = self.model(img_tensor)
                probs = paddle.nn.functional.softmax(output).numpy()[0]
            
            # 解析结果
            return self._parse_result(probs)
            
        except Exception as e:
            logger.error(f"新鲜度检测失败: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"检测失败: {str(e)}"
            }
    
    def _decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """解码图片"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"图片解码失败: {e}")
            return None
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        图片预处理
        - 调整大小到 224x224 (MobileNetV3 的标准输入尺寸)
        - 转换为 RGB
        - 归一化
        """
        # 调整大小
        img = cv2.resize(img, (224, 224))
        
        # 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化 (ImageNet 标准) 并确保使用 float32
        img = img.astype(np.float32)  # 关键修复：显式转换为 float32
        img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # 确保均值和标准差也是 float32
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # 转换为 CHW 格式
        img = img.transpose((2, 0, 1))
        
        return img
    
    def _parse_result(self, probs: np.ndarray) -> Dict[str, any]:
        """解析预测结果"""
        if len(probs) != len(self.label_list):
            return {
            "status": "error",
            "message": f"模型输出类别数 {len(probs)} 与 label_list 长度 {len(self.label_list)} 不一致，请检查模型结构和标签设置"
        }
        # 获取最高分的预测
        top_idx = np.argmax(probs)
        top_label = self.label_list[top_idx]
        top_score = float(probs[top_idx])
        
        # 构建所有结果
        all_results = []
        for i, score in enumerate(probs):
            all_results.append({
                "label": self.label_list[i],
                "score": float(score)
            })
        
        # 按分数排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "status": "success",
            "label": top_label,
            "score": top_score,
            "all_results": all_results
        }
    
    def get_freshness_advice(self, label: str, score: float) -> str:
        """
        根据检测结果给出建议
        
        Args:
            label: 新鲜度标签
            score: 置信度分数
            
        Returns:
            建议文本
        """
        advice_map = {
            "新鲜": "食品状态良好，可以放心食用。建议尽快食用以保持最佳口感。",
            "一般": "食品状态一般，建议尽快食用。注意检查是否有异味或变色。",
            "变质": "食品已变质，不建议食用，请妥善处理。",
            "轻微变质": "食品可能开始变质，建议仔细检查后决定是否食用。",
            "严重变质": "食品严重变质，请立即丢弃，避免食用。"
        }
        
        advice = advice_map.get(label, "请根据实际情况判断是否食用。")
        
        # 如果置信度较低，添加提醒
        if score < 0.7:
            advice += " （注意：检测置信度较低，建议人工再次确认）"
        
        return advice