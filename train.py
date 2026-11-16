from ultralytics import YOLO
m = YOLO("/tmp/claude-0/weights/yolov8n.pt")
m.train(data="/tmp/claude-0/data/pcb_tiles/data_fast.yaml", imgsz=640, epochs=6, batch=16, workers=4, device="cpu",
        project="/tmp/claude-0/runs", name="pcb_yolov8n", plots=False, patience=100, seed=0, exist_ok=True,
        mosaic=0.5, close_mosaic=2, verbose=True)
