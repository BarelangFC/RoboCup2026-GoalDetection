"""
diagnose.py — Run one frame of inference and print all detections.

Usage: python3 diagnose.py
Saves a frame to /tmp/diag_frame.jpg and prints detection scores.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np
from core.trt_inference import TRTYoloDetector

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed")
    sys.exit(1)

# Grab a frame
ret, frame = cap.read()
cap.release()
if not ret:
    print("No frame")
    sys.exit(1)

cv2.imwrite("/tmp/diag_frame.jpg", frame)
print(f"Frame: {frame.shape}")

# Run detection
detector = TRTYoloDetector("/home/nano/yolo26n.engine", conf_threshold=0.05, nms_threshold=0.5)
detector.load()

# Also do raw inference to see all outputs
import tensorrt as trt
import ctypes

# Raw analyse: run preprocess and check output
input_w = 640
input_h = 640
h, w = frame.shape[:2]
scale = min(input_w / w, input_h / h)
new_w = int(w * scale)
new_h = int(h * scale)
resized = cv2.resize(frame, (new_w, new_h))
pad_x = (input_w - new_w) // 2
pad_y = (input_h - new_h) // 2
canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
blob = canvas.astype(np.float32) / 255.0
blob = np.transpose(blob, (2, 0, 1))
blob = np.expand_dims(blob, axis=0)
blob = blob.ravel()

# Run TRT
from core.trt_inference import cuda_mem_alloc, cuda_memcpy_htod, cuda_memcpy_dtoh, cuda_stream_create, cuda_stream_synchronize

stream = detector._stream
d_input = detector._d_input
d_outputs = detector._d_outputs
h_outputs = detector._h_outputs

cuda_memcpy_htod(d_input, blob)
bindings = [int(d_input)] + [int(d) for d in d_outputs]
detector._context.execute_async_v2(bindings, stream, None)
for h_out, d_out in zip(h_outputs, d_outputs):
    cuda_memcpy_dtoh(h_out, d_out)
cuda_stream_synchronize(stream)

# Parse output
output = h_outputs[0]
total = output.size
print(f"\nRaw output size: {total}")

output = output.reshape(1, 8400, 84)[0]
print(f"Reshaped: {output.shape}")

# Find top detections
all_scores = []
for i, pred in enumerate(output):
    scores = pred[4:]
    max_score = float(np.max(scores))
    max_idx = int(np.argmax(scores))
    if max_score > 0.05:
        all_scores.append((max_score, max_idx, i))

all_scores.sort(reverse=True)
print(f"\nTop 20 detections (score > 0.05):")
for score, cls, idx in all_scores[:20]:
    print(f"  Prediction {idx}: class {cls} score={score:.4f}")

# Specifically check class 32 (sports ball)
ball_scores = [(float(p[4+32]), i) for i, p in enumerate(output) if float(p[4+32]) > 0.05]
ball_scores.sort(reverse=True)
print(f"\nBall (class 32) detections: {len(ball_scores)}")
for score, idx in ball_scores[:5]:
    print(f"  Prediction {idx}: score={score:.4f}")

# Also check classes around 32
for cls_check in [29, 30, 31, 32, 33, 34, 35]:
    scores = [float(p[4+cls_check]) for p in output]
    max_s = max(scores)
    if max_s > 0.1:
        print(f"Class {cls_check} max score: {max_s:.4f}")

detector.unload()
print("\nDone")
