import paddleocr, paddle
print("OCR:", paddleocr.__version__)
print("Paddle:", paddle.__version__)


from paddleocr import PaddleOCR
ocr = PaddleOCR(use_textline_orientation=True)  # use_gpu=True启用GPU
image_path = "D:/Smart_Food_Manager/tests/images/test.png"
result = ocr.predict(image_path)
for line in result:
    line.print()