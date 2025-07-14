#from fastapi import FastAPI

#app = FastAPI()

#@app.get("/")
#def read_root():
#    return {"msg": "Hello, Smart Food Manager!"}
#-------------------------------------------------------------------------------------
#启动命令uvicorn app.main:app --reload
# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.backend.ocr import do_ocr

app = FastAPI(title="智能食品管家 OCR 服务")

@app.get("/")
def home():
    return {"msg": "Welcome to Smart Food Manager!"}

@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    # 1) 简单校验
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")
    # 2) 读取二进制
    img_bytes = await file.read()
    # 3) 调用 OCR
    try:
        text = do_ocr(img_bytes)
        return {"text": text}
    except Exception as e:
        # 出错时返回 500
        raise HTTPException(500, f"OCR 处理失败：{e}")
    
'''--------------------------------------------------------------------------------------'''
    # app/main.py 或 app/routers/freshness.py
from app.backend.freshness_detector import FreshnessDetector

# 初始化新鲜度检测器（全局单例）
freshness_detector = FreshnessDetector(
    model_name="ResNet50_vd",  # 或使用你的自定义模型路径
    label_list=["新鲜", "一般", "变质"],
    device="gpu"
)

@app.post("/api/freshness")
async def detect_freshness(file: UploadFile = File(...)):
    """
    食品新鲜度检测接口
    
    Returns:
        {
            "status": "success",
            "label": "新鲜",
            "score": 0.95,
            "advice": "食品状态良好，可以放心食用。",
            "all_results": [...]
        }
    """
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件")
    
    # 读取图片
    img_bytes = await file.read()
    
    # 执行检测
    result = freshness_detector.predict(img_bytes)
    
    if result["status"] == "error":
        raise HTTPException(500, result["message"])
    
    # 添加建议
    advice = freshness_detector.get_freshness_advice(
        result["label"], 
        result["score"]
    )
    result["advice"] = advice
    
    return result