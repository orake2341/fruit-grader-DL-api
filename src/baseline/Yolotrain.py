from ultralytics import YOLO

# ✅ Load YOLOv3 model (pretrained on COCO)
model = YOLO("../../models/yolov3.pt")  # Uses the official YOLOv3 weights

# ✅ Train the model
results = model.train(
    data="C:/Users/CudImon/source/repos/orake2341/fruit-grader-DL-api/fruit-grader-DL-api/data/Dataset/data.yaml",  # Path to dataset config
    epochs=30,  # Reduce epochs to speed up training
    batch=32,  # Lower batch size to fit within 4GB VRAM
    imgsz=416,  # Reduce image size to reduce memory usage
    workers=0,  # Reduce CPU workers to balance load
     amp=True,        # Enable Automatic Mixed Precision (reduces memory usage)
    optimizer="AdamW", # Faster convergence than default SGD
    freeze=0,        # No frozen layers (train full model)
    lr0=0.01,        # Higher learning rate for faster training
    momentum=0.937,  # Default momentum works well
    weight_decay=0.0005,  # Helps regularize model
)

# ✅ Save trained model
model.export(format="onnx")  # Exports model for deployment (optional)
print("🎉 Training completed! Best model saved in 'runs/train/'")