import sys; sys.path.insert(0, r"E:\Python\ultralytics")
from ultralytics import YOLO
import os

# ============ 配置区 ============
DATA_YAML = r"E:\YOLO-Learning\YOLOSafety\Construction Site Safety.v30-raw-images_latestversion.yolov8\data.yaml"
PRETRAINED_MODEL = "E:/Python/ultralytics/YOLO-Weight/yolov8n.pt"     # n=最快 s=均衡 m=精度 l=高精度 x=最高
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16                      # RTX 3070 8GB 可以跑 16
DEVICE = 0                           # GPU 训练
PROJECT = r"E:\YOLO-Learning\YOLOSafety\runs"
EXPERIMENT_NAME = "safety_train"

# ============ 训练 ============
if __name__ == "__main__":
    print(f"数据集: {DATA_YAML}")
    print(f"预训练模型: {PRETRAINED_MODEL}")
    print(f"设备: GPU (CUDA)")
    print("=" * 50)

    model = YOLO(PRETRAINED_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=EXPERIMENT_NAME,
        patience=10,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print("=" * 50)
    print(f"训练完成！最佳模型: {results.save_dir / 'weights' / 'best.pt'}")
