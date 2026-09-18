from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os

app = Flask(__name__)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# YOLO11n model path
MODEL_PATH = BASE_DIR / "models" / "pcb_yolo11n" / "weights" / "best.pt"

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


# --------------------------------------------------
# Allowed image formats
# --------------------------------------------------

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# YOLO11n class names
# --------------------------------------------------

class_names = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper"
}


# --------------------------------------------------
# Load YOLO11n model
# --------------------------------------------------

model = YOLO(str(MODEL_PATH))


# --------------------------------------------------
# Image Upload Prediction
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "file" not in request.files:
            return render_template(
                "index.html",
                error="No image file selected."
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image."
            )

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error="Please upload JPG, JPEG, or PNG image."
            )

        filename = secure_filename(file.filename)

        upload_path = UPLOAD_FOLDER / filename

        file.save(upload_path)

        # YOLO11n prediction - CPU friendly for Render
        results = model.predict(
            source=str(upload_path),
            imgsz=320,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append({
                "defect": class_names[class_id],
                "confidence": round(confidence * 100, 2)
            })

        # PASS / FAIL
        if len(detections) == 0:
            status = "PASS"
        else:
            status = "FAIL"

        return render_template(
            "index.html",
            image_path=f"/static/uploads/{filename}",
            detections=detections,
            status=status
        )

    return render_template("index.html")


# --------------------------------------------------
# Web Camera Prediction
# --------------------------------------------------

@app.route("/camera_predict", methods=["POST"])
def camera_predict():

    if "camera_image" not in request.files:
        return {
            "error": "No camera image received."
        }, 400

    file = request.files["camera_image"]

    if file.filename == "":
        return {
            "error": "No camera image selected."
        }, 400

    # Save captured webcam image
    filename = "webcam_capture.jpg"

    upload_path = UPLOAD_FOLDER / filename

    file.save(upload_path)

    # YOLO11n prediction - CPU friendly for Render
    results = model.predict(
        source=str(upload_path),
        imgsz=320,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "defect": class_names[class_id],
            "confidence": round(confidence * 100, 2)
        })

    # PASS / FAIL
    if len(detections) == 0:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "status": status,
        "detections": detections
    }


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
