#from fastapi import FastAPI

#app = FastAPI()

#@app.get("/")
#def read_root():
#    return {"msg": "Hello, Smart Food Manager!"}
#-------------------------------------------------------------------------------------

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