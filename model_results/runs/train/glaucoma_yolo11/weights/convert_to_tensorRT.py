from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO("best.pt")

# Export to TensorRT format
model.export(format="engine")