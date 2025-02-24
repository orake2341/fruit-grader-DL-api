from ultralytics import YOLO

# ✅ Load YOLOv3 model (pretrained on COCO)
model = YOLO("../../models/yolov3.pt")  # Uses the official YOLOv3 weights

# ✅ Train the model
results = model.train(
    data="../../data/Dataset/data.yaml",  # Path to dataset config
    epochs=30,  # Reduce epochs to speed up training
    batch=8,  # Lower batch size to fit within 4GB VRAM
    imgsz=416,  # Reduce image size to reduce memory usage
    workers=0,  # Reduce CPU workers to balance load
    device="cuda",  # Use GPU for acceleration
    augment=False,
    freeze=10,
)

# ✅ Save trained model
model.export(format="onnx")  # Exports model for deployment (optional)
print("🎉 Training completed! Best model saved in 'runs/train/'")
