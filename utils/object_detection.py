import cv2
import numpy as np

# Load class names
with open('models/coco.names', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Load YOLOv4-tiny model
config_path = 'models/yolov4-tiny.cfg'
weights_path = 'models/yolov4-tiny.weights'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Use CPU (you can use DNN_TARGET_CUDA if GPU available)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get output layer names
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # Typical OpenCV format
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]    # Flattened format fallback


def detect_objects(frame, conf_threshold=0.5, nms_threshold=0.4):
    h, w = frame.shape[:2]

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    detected_object_names = []

    # Parse outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x, center_y, width, height = (detection[0:4] * np.array([w, h, w, h])).astype("int")
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
     for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w_box, h_box = boxes[i]
        label = f"{class_names[class_ids[i]]}: {int(confidences[i] * 100)}%"
        detected_object_names.append(class_names[class_ids[i]])

        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    return frame, detected_object_names
