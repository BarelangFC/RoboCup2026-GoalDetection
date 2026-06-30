"""
dump_raw_bbox.py — Run this with a ball in frame, prints raw bbox values.
Run: python3 src/dump_raw_bbox.py
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.trt_inference import TRTYoloDetector
from core.trt_inference import cuda_memcpy_htod, cuda_memcpy_dtoh, cuda_stream_synchronize

d = TRTYoloDetector("/home/nano/yolo26_custom.engine", conf_threshold=0.01)
d.load()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("NO CAMERA")
    sys.exit(1)

for i in range(5):
    ret, frame = cap.read()
    if not ret: continue
cap.release()

blob, scale, pad = d._preprocess(frame, 640, 640)
cuda_memcpy_htod(d._d_input, blob)
bindings = [int(d._d_input)] + [int(dd) for dd in d._d_outputs]
d._context.execute_async_v2(bindings, d._stream, None)
for h_out, d_out in zip(d._h_outputs, d._d_outputs):
    cuda_memcpy_dtoh(h_out, d_out)
cuda_stream_synchronize(d._stream)

output = d._h_outputs[0].reshape(8400, 15)

# Find top ball predictions
scores = output[:, 4]
top = np.argsort(scores)[::-1][:5]
print(f"Frame: {frame.shape}")
print(f"Scale: {scale}, Pad: {pad}")
print(f"Top detections (class 0 = ball):")
for idx in top:
    s = float(scores[idx])
    if s < 0.01: break
    v = [float(output[idx, i]) for i in range(4)]
    print(f"  [{idx}] score={s:.3f} raw=[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f},{v[3]:.2f}]")
    # Try as [x1,y1,x2,y2] absolute
    x1 = (v[0] - pad[0]) / scale
    y1 = (v[1] - pad[1]) / scale
    x2 = (v[2] - pad[0]) / scale
    y2 = (v[3] - pad[1]) / scale
    print(f"    x1y1x2y2: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
    # Try as normalized [0,1] * 640
    x1n = v[0] / (640 if max(v) > 1 else 1)
    y1n = v[1] / (640 if max(v) > 1 else 1)
    x2n = v[2] / (640 if max(v) > 1 else 1)
    y2n = v[3] / (640 if max(v) > 1 else 1)
    print(f"    normalized: [{x1n:.2f},{y1n:.2f},{x2n:.2f},{y2n:.2f}]")
    # Try as cxcywh
    cx = (v[0] - pad[0]) / scale
    cy = (v[1] - pad[1]) / scale
    bw = v[2] / scale
    bh = v[3] / scale
    print(f"    cxcywh: [{cx-bw/2:.1f},{cy-bh/2:.1f},{cx+bw/2:.1f},{cy+bh/2:.1f}]")

d.unload()
print("Done. Send the raw values above.")
