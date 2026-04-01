import threading
import time
import io
import json

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from flask import Flask, Response, jsonify, render_template, stream_with_context

app = Flask(__name__)

# ======= Load model & labels (sesuaikan nama file) =======
MODEL_PATH = "model_statis_dense.keras"
LABELS_PATH = "label_encoder_keras.npy"

model = load_model(MODEL_PATH)
classes = np.load(LABELS_PATH, allow_pickle=True)

# ======= MediaPipe setup =======
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ======= Shared state between threads =======
output_frame = None        # JPEG bytes
latest_label = ""
latest_conf = 0.0
lock = threading.Lock()

# ======= Camera processing loop (background thread) =======
def camera_loop(src=0):
    global output_frame, latest_label, latest_conf, hands
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError("Kamera tidak terdeteksi. Periksa index kamera atau akses.")

    # ====== Tambahan: Samakan resolusi dengan Program 1 ======
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)  # beri waktu kamera stabil

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # ====== Konversi untuk MediaPipe ======
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label = ""
        conf = 0.0

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                ).flatten().reshape(1, -1)

                prediction = model.predict(coords, verbose=0)
                idx = int(np.argmax(prediction))
                label = str(classes[idx])
                conf = float(np.max(prediction))

                cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hands", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ====== JPEG encode ======
        # (Opsional: ubah kualitas untuk mengurangi noise)
        ret2, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret2:
            continue

        with lock:
            output_frame = jpeg.tobytes()
            latest_label = label
            latest_conf = conf

        # ====== Stabilkan FPS (30–33 FPS) ======
        time.sleep(0.03)

    cap.release()

# start camera thread when app starts
threading.Thread(target=camera_loop, args=(0,), daemon=True).start()

# ======= Video feed route (MJPEG) =======
def generate_mjpeg():
    boundary = b'--frame\r\n'
    while True:
        with lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary + b'Content-Type: image/jpeg\r\nContent-Length: ' + f"{len(frame)}".encode() + b'\r\n\r\n' + frame + b'\r\n'
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_mjpeg()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ======= Label stream route (Server-Sent Events) =======
@app.route('/label_stream')
def label_stream():
    def event_stream():
        last_sent = ""
        while True:
            with lock:
                lab = latest_label
                conf = latest_conf
            payload = {"label": lab, "confidence": float(conf)}
            # only send when changed or periodically
            data = json.dumps(payload)
            if data != last_sent:
                yield f"data: {data}\n\n"
                last_sent = data
            else:
                # keep connection alive
                yield ": keep-alive\n\n"
            time.sleep(0.15)
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

@app.route('/predict_label')
def predict_label():
    """
    Endpoint untuk ESP8266: mengembalikan huruf terakhir
    yang terdeteksi dari kamera.
    """
    with lock:
        label = latest_label
        confidence = latest_conf
    return jsonify({"label": label, "confidence": float(confidence)})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Gunakan host='0.0.0.0' bila ingin akses dari perangkat lain
    app.run(host='0.0.0.0', port=5000, threaded=True)

