import cv2
from ultralytics import YOLO  # 导入YOLO目标检测模型

# 加载训练好的模型权重文件
model = YOLO(r"E:/YOLO-Learning/YOLOSafety/runs/safety_train/weights/best.pt")

# 打开视频文件（也可以改成摄像头 0）
# cap = cv2.VideoCapture("E:/YOLO-Learning/data_shouji/ppe-2-1.mp4")

#用摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)



# 循环读取视频帧
while True:

    # 读取一帧图像
    success, img = cap.read()

    # 如果读取失败（视频结束），退出循环
    if not success:
        break

    # 使用YOLO模型对当前帧进行检测
    results = model(img)

    # 获取检测框结果
    boxes = results[0].boxes

    # 遍历每一个检测目标
    for box in boxes:

        # 获取边界框坐标（xyxy格式）
        x1, y1, x2, y2 = box.xyxy[0]

        # 转换为整数（用于OpenCV绘图）
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # 获取类别ID和置信度
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # 根据类别ID获取类别名称
        class_name = model.names[cls]

        # 根据类别设置颜色和标签
        if class_name == "NO-Mask":
            color = (0, 0, 255)
            label = f"ALERT: {class_name} {conf:.2f}"

        elif class_name == "NO-Hardhat":
            color = (0, 0, 255)
            label = f"ALERT: {class_name} {conf:.2f}"

        elif class_name == "NO-Safety Vest":
            color = (0, 0, 255)
            label = f"ALERT: {class_name} {conf:.2f}"

        elif class_name in ["Mask", "Hardhat", "Safety Vest"]:
            color = (0, 255, 0)
            label = f"OK: {class_name} {conf:.2f}"

        else:
            color = (255, 0, 0)
            label = f"{class_name} {conf:.2f}"

        # 画检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 在框上方写文字
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # 显示结果窗口
    cv2.imshow("Safety Detection", img)

    # 按ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放视频资源
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()