import os
import cv2
from flask import Flask, render_template, request, flash, redirect, Response
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from collections import Counter
from werkzeug.utils import secure_filename

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
print("Loading YOLOv8 model...")
model = YOLO('best.pt')  # Ensure this file exists in your working directory
print("Model loaded successfully.")

# -----------------------------
# Home Page: Upload Image and Run Detection
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file or file.filename == '':
                flash("No file uploaded.")
                return redirect(request.url)

            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            # Validate image file
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
            output_path = os.path.join(OUTPUT_FOLDER, filename)
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

        except Exception as e:
            print(f"Error during detection: {e}")
            flash("An error occurred during detection.")
            return redirect(request.url)

    return render_template('index.html')

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

        results = model(frame)
        result_img = results[0].plot()

        _, buffer = cv2.imencode('.jpg', result_img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# Run the app (Render-compatible)
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
