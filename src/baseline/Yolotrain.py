from ultralytics import YOLO

# ✅ Load YOLOv3 model (pretrained on COCO)
model = YOLO("../../models/yolov5m.pt")  # Uses the official YOLOv3 weights

# ✅ Train the model
results = model.train(
    data="../../data/Dataset/data.yaml",  # Path to dataset config
    epochs=50,
    batch=8,
    imgsz=416,  # Reduce image size to reduce memory usage
    workers=0,  # Reduce CPU workers to balance load
    amp=True,  # Enable Automatic Mixed Precision (reduces memory usage)
    optimizer="Adam",  # Faster convergence than default SGD
    freeze=0,
    lr0=0.01,  # learning rate
    weight_decay=0.0005,  # Helps regularize model
)

# ✅ Save trained model
model.export(format="onnx")  # Exports model for deployment (optional)
print("🎉 Training completed! Best model saved in 'runs/train/'")
