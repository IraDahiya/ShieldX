from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import psutil
import time
from utils.object_detection import detect_objects

app = Flask(__name__)
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)

fps = 0
prev_time = 0

detection_count = 0

def get_network_status():
    # Basic network check example (replace with actual ping or status check)
    net_io = psutil.net_io_counters()
    return {
        'bytes_sent': net_io.bytes_sent,
        'bytes_recv': net_io.bytes_recv
    }

def generate_frames():
    global fps, prev_time, detection_count
    while True:
        success, frame = cap.read()
        if not success:
            break

        annotated_frame, detected_objects = detect_objects(frame)
        detection_count = len(detected_objects)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@socketio.on('connect')
def send_metrics():
    def metrics_loop():
        while True:
            cpu = psutil.cpu_percent(interval=1)
            net = get_network_status()
            # Emit metrics
            socketio.emit('metrics', {
                'fps': round(fps, 2),
                'cpu': cpu,
                'detection_count': detection_count,
                'bytes_sent': net['bytes_sent'],
                'bytes_recv': net['bytes_recv']
            })
    socketio.start_background_task(metrics_loop)

@app.route('/')
def index():
    return render_template('index.html')  # We'll create this next

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
