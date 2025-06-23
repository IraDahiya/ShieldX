import cv2

config_path = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
model_path = "models/frozen_inference_graph.pb"

try:
   net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
   print("Model loaded with readNetFromTensorflow")

except cv2.error as e:
    print("OpenCV error:", e)
except Exception as e:
    print("General error:", e)
