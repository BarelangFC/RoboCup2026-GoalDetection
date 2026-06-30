"""
debug_bbox.py — Dump raw bbox values from the custom model.
"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core.trt_inference import TRTYoloDetector
from core.trt_inference import cuda_memcpy_htod, cuda_memcpy_dtoh, cuda_stream_synchronize

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("NO CAMERA - using blank frame")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
else:
    ret, frame = cap.read()
    cap.release()

cv2.imwrite("/tmp/debug_frame.jpg", frame)
print(f"Frame: {frame.shape}")

# Load detector
d = TRTYoloDetector("/home/nano/yolo26_custom.engine", conf_threshold=0.1)
d.load()

# Preprocess
blob, scale, pad = d._preprocess(frame, 640, 640)
print(f"Scale: {scale}, Pad: {pad}")
print(f"Input size: {frame.shape[1]}x{frame.shape[0]}")

# Run inference
cuda_memcpy_htod(d._d_input, blob)
bindings = [int(d._d_input)] + [int(dd) for dd in d._d_outputs]
d._context.execute_async_v2(bindings, d._stream, None)
for h_out, d_out in zip(d._h_outputs, d._d_outputs):
    cuda_memcpy_dtoh(h_out, d_out)
cuda_stream_synchronize(d._stream)

output = d._h_outputs[0].reshape(8400, 15)

# Find top ball detections
ball_scores = output[:, 4]  # class 0
top_indices = np.argsort(ball_scores)[::-1][:10]
print(f"\nTop 10 ball detections:")
for idx in top_indices:
    score = float(ball_scores[idx])
    if score < 0.1: break
    v0, v1, v2, v3 = float(output[idx,0]), float(output[idx,1]), float(output[idx,2]), float(output[idx,3])
    print(f"  [{idx}] score={score:.3f}  bbox_raw=[{v0:.2f}, {v1:.2f}, {v2:.2f}, {v3:.2f}]")

# Also try the detection method
results = d.detect(frame)
print(f"\nDetect() returned: {len(results)} results")
for r in results:
    print(f"  bbox={[f'{v:.1f}' for v in r.bbox]} conf={r.confidence:.3f}")

d.unload()
