import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------- Color Ranges (HSV) ----------
COLOR_RANGES = {
    "Red": ([136, 87, 111], [180, 255, 255]),
    "Green": ([25, 52, 72], [102, 255, 255]),
    "Blue": ([94, 80, 2], [120, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255])
}

def detect_colors(frame):
    """Detect colors in the given frame and draw bounding boxes."""
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), "uint8")

    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)
        mask = cv2.inRange(hsvFrame, lower, upper)
        mask = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # ignore small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"{color_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/detect_image", methods=["POST"])
def detect_image():
    """Handle uploaded image detection."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Convert uploaded file to numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Process detection
    processed = detect_colors(img)

    _, buffer = cv2.imencode(".jpg", processed)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": f"data:image/jpeg;base64,{encoded_img}"})

@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    """Handle live camera frame detection from browser."""
    data = request.json["image"]
    image_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed = detect_colors(frame)

    _, buffer = cv2.imencode(".jpg", processed)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": f"data:image/jpeg;base64,{encoded_img}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
