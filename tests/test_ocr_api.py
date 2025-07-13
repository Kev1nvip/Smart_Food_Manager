# D:\Smart_Food_Manager\tests\test_ocr_api.py

import pytest
from fastapi.testclient import TestClient
# 从你的 main.py 文件中导入 FastAPI 应用实例
# 这里的路径是相对于项目的根目录 (`D:\Smart_Food_Manager\`) 来看的
from app.main import app

# 创建 TestClient 实例
client = TestClient(app)

# 定义测试数据：一个包含多个测试用例的列表
# 每个元组包含：测试图片的路径, 预期在识别结果中出现的文本片段
# 请根据你放在 tests/images/ 目录下的实际图片和图片中的文本来修改这个列表
test_cases = [
    ("tests/images/test.png", "水、食用盐"), # 替换为你实际的图片和其中的文字
    #("./tests/images/another_test.png", "重要信息"), # 如果你有第二张测试图
    # 可以添加更多测试用例
    # ("./tests/images/blurred_text.jpg", "模糊的文字可能无法识别"), # 测试边缘情况
]

# 使用 pytest 的 parametrize 装饰器，让同一个测试函数可以运行多次，每次使用 test_cases 中的一个元组
@pytest.mark.parametrize("image_path, expected_text_part", test_cases)
def test_ocr_endpoint(image_path: str, expected_text_part: str):
    """
    测试 /ocr/ 端点：上传图片并验证 OCR 结果是否包含预期文本。
    """
    # 确保测试图片文件存在
    try:
        with open(image_path, "rb") as f:
            # TestClient 需要 files 参数的格式是 {"文件名": (实际文件名, 文件内容 bytes, MIME类型)}
            files = {"file": (image_path.split("/")[-1], f.read(), "image/jpeg/png")} # MIME类型可以根据实际图片类型调整，jpeg/png/webp等
    except FileNotFoundError:
        pytest.fail(f"测试图片文件未找到: {image_path}")

    # 使用 TestClient 发起 POST 请求到 /ocr/ 端点
    response = client.post("/ocr/", files=files)

    # 验证 HTTP 状态码是否为 200 (成功)
    assert response.status_code == 200, f"期望状态码 200，实际是 {response.status_code}，响应体: {response.text}"

    # 解析 JSON 响应体
    data = response.json()

    # 验证响应体是字典，并且包含 'text' 键
    assert isinstance(data, dict), "响应体不是字典"
    assert "text" in data, "响应体中缺少 'text' 键"

    # 验证 'text' 键的值是字符串
    recognized_text = data["text"]
    assert isinstance(recognized_text, str), "'text' 的值不是字符串"

    # 验证识别出的文本中包含了我们期望的文本片段
    assert expected_text_part in recognized_text, f"OCR 识别结果未包含 '{expected_text_part}'。\n完整识别结果:\n{recognized_text}"

# 可以添加一个测试，验证上传非图片文件时的错误处理
def test_ocr_endpoint_invalid_file_type():
    """
    测试上传非图片文件时是否返回 400 错误。
    """
    # 模拟一个非图片文件
    files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    response = client.post("/ocr/", files=files)

    # 验证状态码是否为 400
    assert response.status_code == 400, f"期望非图片文件返回 400，实际是 {response.status_code}"
    data = response.json()
    # 验证错误详情
    assert data.get("detail") == "请上传图片文件"


# 可以添加一个测试，验证没有上传文件时的行为 (FastAPI 默认会返回 422 Validation Error)
def test_ocr_endpoint_no_file():
    """
    测试不上传文件时是否返回 422 验证错误。
    """
    response = client.post("/ocr/")
    # 验证状态码是否为 422 (Unprocessable Entity)
    assert response.status_code == 422, f"期望未上传文件返回 422，实际是 {response.status_code}"
    # FastAPI 默认的 422 响应体格式，通常包含 errors 列表
    data = response.json()
    assert "detail" in data
    assert isinstance(data["detail"], list)
    assert len(data["detail"]) > 0
    assert data["detail"][0].get("type") == "missing" # 检查错误类型是否为缺少参数
