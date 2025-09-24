from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Color detection logic (update as per your needs)
def detect_colors(imageFrame):
    hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Example: detect red and blue colors
    colors_detected = []

    # Red range
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    if cv2.countNonZero(red_mask) > 500:
        colors_detected.append("Red")

    # Blue range
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    if cv2.countNonZero(blue_mask) > 500:
        colors_detected.append("Blue")

    return colors_detected

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/process_frame", methods=["POST"])
def process_frame():
    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame received"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    imageFrame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    colors = detect_colors(imageFrame)

    return jsonify({"colors": colors})

if __name__ == "__main__":
    app.run(debug=True)
