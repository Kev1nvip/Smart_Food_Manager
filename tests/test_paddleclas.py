# test_paddleclas.py
import os
os.environ['PPIM_CACHE_HOME'] = 'C:/Users/CC/.paddleclas'  # 自定义路径
os.environ['PADDLE_DOWNLOAD_HOST'] = 'https://paddleclas.bj.bcebos.com'

try:
    from paddleclas import PaddleClas
    print("导入成功")
    
    # 用 CPU 测试
    clas = PaddleClas(model_name="ResNet50_vd", device="cpu")
    print("初始化成功")
    
    # 测试预测（用随机数据）
    import numpy as np
    fake_img = np.random.rand(224, 224, 3).astype('uint8')
    result = clas.predict(fake_img)
    print("预测成功:", result)
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()