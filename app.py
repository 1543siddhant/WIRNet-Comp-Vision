import os
import cv2
from PIL import Image
from flask import Flask, request, render_template, url_for
from dotenv import load_dotenv

from ultralytics import YOLO
from utils import compute_custom_path
from ocr_tool import OCRTool

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv8 model and OCR tool once
model = YOLO("best.pt")
ocr   = OCRTool()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # 1) Save upload
    f       = request.files["image"]
    in_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(in_path)

    # 2) Run detection
    img     = cv2.imread(in_path)
    results = model.predict(img, conf=0.25)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append({
            "label": model.names[cls],
            "coords": (cx, cy),
            "bbox":   (x1, y1, x2, y2)
        })

    # 3) Identify main, switch, walls
    main_pt   = next(d["coords"] for d in detections if d["label"] == "main")
    switch_pt = next(d["coords"] for d in detections if d["label"] == "switch")
    walls     = [d for d in detections if d["label"] == "wall"]

    # 4) Compute path using rotation‐invariant right‐side logic
    path_nodes, pixel_len = compute_custom_path(main_pt, switch_pt, walls)

    # 5) Convert to feet (1cm = 1ft; adjust as needed)
    pixels_per_cm = 10
    length_ft     = pixel_len / pixels_per_cm

    # 6) Draw path and endpoints
    vis = img.copy()
    pts = [(int(x), int(y)) for x, y in path_nodes]
    for i in range(len(pts) - 1):
        cv2.line(vis, pts[i], pts[i+1], (0, 0, 255), 3)
    cv2.circle(vis, (int(main_pt[0]), int(main_pt[1])), 6, (0, 255, 0), -1)
    cv2.circle(vis, (int(switch_pt[0]), int(switch_pt[1])), 6, (255, 0, 0), -1)

    out_name = "result.png"
    out_path = os.path.join(RESULT_FOLDER, out_name)
    cv2.imwrite(out_path, vis)
    img_url = url_for("static", filename=f"results/{out_name}")

    # 7) OCR text extraction
    pil  = Image.open(in_path)
    text = ocr.from_pytesseract(pil) or ocr.from_ocr_space(in_path)

    # 8) Render results inline
    return render_template(
        "index.html",
        length_ft    = round(length_ft, 2),
        waypoints    = path_nodes,
        ocr_text     = text,
        result_image = img_url
    )

if __name__ == "__main__":
    app.run(debug=True)
