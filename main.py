import os
import cv2
from datetime import datetime
from utils.object_detection import detect_objects

# Only save snapshots and alert for these classes
FILTER_CLASSES = {"person", "remote", "notebook", "pen", "bottle", "cell phone", "cellphone", "phone"}

os.makedirs('captures', exist_ok=True)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, detected_classes = detect_objects(frame)

        # Check if any detected class matches FILTER_CLASSES (case insensitive)
        matched = [cls for cls in detected_classes if cls.lower() in FILTER_CLASSES]

        if matched:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'captures/detection_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Alert! Detected: {matched} - Snapshot saved: {filename}")

        cv2.imshow("ShieldX - Filtered Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
