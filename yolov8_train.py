from ultralytics import YOLO

if __name__ == "__main__":
    # Load a small pretrained model for fast iteration
    model = YOLO("yolov8n.pt")

    # Train on your custom dataset
    model.train(
        data="data.yaml",
        epochs=110,
        imgsz=640,
        batch=16,
        workers=4,
        project="runs",           # output folder
        name="detect_floorplan"   # run name
    )

    print("Training complete! Weights saved in runs/detect_floorplan/weights/best.pt")
