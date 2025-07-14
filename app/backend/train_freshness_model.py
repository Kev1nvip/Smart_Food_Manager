# train_freshness_model.py
from paddleclas import PaddleClasTrainer

# 准备数据集（目录结构）
# dataset/
#   train/
#     fresh/
#     normal/
#     spoiled/
#   val/
#     fresh/
#     normal/
#     spoiled/

trainer = PaddleClasTrainer(
    model_name="ResNet50_vd",
    dataset_dir="./dataset",
    num_classes=3,
    epochs=20,
    batch_size=32
)

trainer.train()
trainer.export_model("./models/freshness_model")