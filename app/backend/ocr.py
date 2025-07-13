# app/ocr.py
import logging
from paddleocr import PaddleOCR
import numpy as np
import cv2

logger = logging.getLogger(__name__)
logger.info("正在初始化 PaddleOCR 引擎...")
ocr_engine = PaddleOCR(lang='ch')
# 初始化 OCR 引擎（优先用 GPU，失败则用 CPU）

try:
    ocr_engine = PaddleOCR(lang='ch', device='gpu')
    logger.info("PaddleOCR 引擎初始化成功（GPU）")
except Exception as e:
    logger.warning(f"PaddleOCR GPU 初始化失败，尝试使用 CPU: {e}")
    try:
        ocr_engine = PaddleOCR(lang='ch', device='cpu')
        logger.info("PaddleOCR 引擎初始化成功（CPU）")
    except Exception as cpu_e:
        logger.error(f"PaddleOCR CPU 初始化也失败: {cpu_e}")
        ocr_engine = None  # 明确标记初始化失败


def do_ocr(image_bytes: bytes) -> str:
    """
    输入：图像二进制
    返回：识别到的文字（按行拼接）
    """
    if ocr_engine is None:
        logger.error("OCR 引擎未成功初始化，无法执行 OCR")
        raise RuntimeError("OCR 引擎未就绪")

    try:
        # PaddleOCR 3.x 推荐用 ocr() 方法
        nparr = np.frombuffer(image_bytes, np.uint8)
        # 2. 解码为 OpenCV 图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #print("img.shape:", img.shape)
        #cv2.imwrite("debug.jpg", img)  # 保存解码后的图片，手动检查
        result = ocr_engine.predict(img)
        #print("result:", result)
        lines = []
        # 解析结果
        if result and isinstance(result, list) and 'rec_texts' in result[0]:
            lines = result[0]['rec_texts']
        # 老版兼容
        elif result and isinstance(result, list) and len(result) > 0:
            for line in result[0]:
                text = line[1][0]
                lines.append(text)
        else:
            logger.warning(f"OCR 识别结果为空或格式异常: {result}")
        logger.info(f"PaddleOCR 识别完成，识别到 {len(lines)} 行文本")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"do_ocr 函数内部发生错误: {e}", exc_info=True)
        raise e