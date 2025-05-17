import os
import cv2
from flask import Flask, render_template, request, flash, redirect, Response
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# -----------------------------
# Folder Setup
# -----------------------------
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model = YOLO('best.pt')  # Update if needed

# -----------------------------
# Home Page: Upload Image
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    if not file or file.filename == '':
        flash("No file uploaded.")
        return redirect(request.url)

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Validate image
    try:
        with Image.open(img_path) as img:
            img.verify()
    except (UnidentifiedImageError, IOError):
        os.remove(img_path)
        flash("Invalid image file. Please upload a valid image.")
        return redirect(request.url)

    # Run YOLOv8 inference
    results = model(img_path)
    result = results[0]
    result_img = result.plot()

    # Save annotated image
    output_path = os.path.join(OUTPUT_FOLDER, file.filename)
    cv2.imwrite(output_path, result_img)

    # Count detected objects
    class_names = model.names
    detected_classes = [class_names[int(cls)] for cls in result.boxes.cls]
    count_dict = dict(Counter(detected_classes))

    return render_template(
        'index.html',
        uploaded_image=output_path,
        counts=count_dict
    )

# -----------------------------
# Webcam Detection (Live Feed)
# -----------------------------
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLOv8 detection
        results = model(frame)
        result_img = results[0].plot()

        # Convert to byte stream
        _, buffer = cv2.imencode('.jpg', result_img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Page to view webcam feed
@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

# Video feed stream
@app.route('/webcam_feed')
def webcam_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)
