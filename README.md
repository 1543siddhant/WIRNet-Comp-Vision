# WIRNet-Comp-Vision

**Wiring Intelligence and Recognition Network for Blueprints**

## Overview

WIRNet-Comp-Vision is a Flask-based computer vision pipeline designed to: 1) detect walls and electrical elements in architectural blueprints using a YOLOv8-oriented bounding box (OBB) model, 2) perform OCR on annotations, and 3) compute and visualize optimal wiring paths (e.g., from a main panel to a switch) based on custom path-planning logic.

Key features:

* **Object Detection**: YOLOv8 OBB model trained to detect walls and electrical fixtures.
* **OCR Integration**: Extract text annotations from blueprints via an OCR API.
* **Custom Routing**: `compute_custom_path` identifies intermediate wall anchor points on the right-hand side to compute a realistic wiring path.
* **Web App Interface**: Upload blueprints via a web UI and receive annotated images, PDF reports, and path visualizations.

## Repository Structure

```
WIRNet-Comp-Vision/
├── app.py              # Flask app: endpoints for upload, inference, and reporting
├── utils.py            # Utility functions (distance, filtering, path computation)
├── ocr_tool.py         # OCRTool class: integrates with external OCR API
├── yolov8_train.py     # Script to train YOLOv8 model with rotated bboxes on .yaml data
├── data.yaml           # YOLOv8 training dataset config
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates (index.html, result.html)
├── static/             # Static assets (CSS, JS, annotated images)
├── uploads/            # Directory for storing uploaded blueprints
├── runs/               # YOLOv8 training outputs
├── best.pt             # Pre-trained YOLOv8 model weights
└── yolov8n.pt          # YOLOv8-nano base weights
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/1543siddhant/WIRNet-Comp-Vision.git
   cd WIRNet-Comp-Vision
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OCR API key**:

   * Copy `.env.example` to `.env` and add your `OCR_SPACE_API_KEY`.

## Usage

### 1. Run the Flask Web App

```bash
python app.py
```

* Navigate to `http://127.0.0.1:5000` in your browser.
* Upload a blueprint image (JPEG/PNG/PDF).
* View detection, OCR results, and custom wiring path.
* Download annotated image or PDF report.

### 2. Training a New YOLOv8 Model

```bash
python yolov8_train.py --data data.yaml --epochs 50 --weights yolov8n.pt --name wiring_obb
```

* Adjust `--epochs` and other hyperparameters as needed.
* Trained weights will be saved under `runs/train/wiring_obb/weights/best.pt`.

## `compute_custom_path` Function

Located in `utils.py`, this function:

1. Detects wall centers from YOLOv8 outputs.
2. Filters walls on the right side relative to the main→switch vector using a rotation-invariant cross-product check.
3. Selects up to three intermediate wall anchor points (`w1`, `w2`, `w3`) in order:

   * `w1`: among the two closest to the main point, pick the one farthest from the switch.
   * `w2`: closest to `w1` among remaining.
   * `w3`: closest to the switch among remaining.
4. Returns the ordered path `[main, w1, w2?, w3?, switch]` and its total length.

```python
path, total_length = compute_custom_path(main_pt, switch_pt, wall_detections)
```

## OCR Integration (`ocr_tool.py`)

* **`OCRTool`** wraps calls to the OCR.space API.
* Extracts detection text and bounding boxes from blueprint annotations.
* Saves OCR outputs to JSON for downstream reporting.

## Reporting

After processing, the app generates:

* **Annotated Image**: Shows detected walls, fixtures, OCR boxes, and routing path.
* **PDF Report**: Combines blueprint, detection tables, OCR text, and path metrics.

## Dependencies

See `requirements.txt` for exact package versions. Major dependencies include:

* Flask
* ultralytics (YOLOv8)
* OpenCV
* Pillow
* requests
* python-dotenv

## Contributing

1. Fork the repo. 2. Create a feature branch: `git checkout -b feature/YourFeature`. 3. Commit changes. 4. Open a Pull Request.

## Contact

For questions or support, please open an issue or contact: `siddhantpatil1543@gmail.com`.

---

