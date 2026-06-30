"""
trt_inference.py — Pure TensorRT inference for YOLO detection.
Custom model support: handles [1, N, 6] NMS-processed output format.
"""

import os, time, ctypes, ctypes.util
import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# CUDA via ctypes (no pycuda needed)
_cuda_lib = None
def _cuda():
    global _cuda_lib
    if _cuda_lib is None:
        p = ctypes.util.find_library("cudart") or "libcudart.so.10.2"
        _cuda_lib = ctypes.cdll.LoadLibrary(p)
    return _cuda_lib

def cuda_mem_alloc(size):
    ptr = ctypes.c_void_p()
    _cuda().cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
    return ptr.value

def cuda_mem_free(ptr):
    _cuda().cudaFree(ctypes.c_void_p(ptr))

def cuda_memcpy_htod(ptr, arr):
    a = np.ascontiguousarray(arr)
    _cuda().cudaMemcpy(ctypes.c_void_p(ptr), a.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_size_t(a.nbytes), ctypes.c_int(1))

def cuda_memcpy_dtoh(arr, ptr):
    a = np.ascontiguousarray(arr)
    _cuda().cudaMemcpy(a.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
                       ctypes.c_size_t(a.nbytes), ctypes.c_int(2))

def cuda_stream_create():
    s = ctypes.c_void_p()
    _cuda().cudaStreamCreate(ctypes.byref(s))
    return s.value

def cuda_stream_synchronize(s):
    _cuda().cudaStreamSynchronize(ctypes.c_void_p(s))


class Detection:
    __slots__ = ("bbox", "confidence", "class_id")
    def __init__(self, bbox, confidence, class_id):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id


class TRTInferenceError(Exception):
    pass


class TRTYoloDetector:
    """TensorRT YOLO detector — handles [1,N,6] and [1,8400,84] outputs."""

    BALL_CLASS_ID = 0        # Custom model: ball is class 0
    OUTPUT_ITEMS = 6          # [x1,y1,x2,y2,conf,cls] per detection

    def __init__(self, engine_path, conf_threshold=0.25, nms_threshold=0.45):
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.bbox_offset_x = 0
        self.bbox_offset_y = 0
        self.bbox_scale_x = 1.0
        self.bbox_scale_y = 1.0
        self._engine = None
        self._context = None
        self._stream = None
        self._d_input = None
        self._d_outputs = []
        self._h_outputs = []
        self._input_shape = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self.class_names = {0: "ball"}

    def load(self):
        if not TRT_AVAILABLE:
            raise TRTInferenceError("tensorrt not available")
        if not os.path.exists(self.engine_path):
            raise TRTInferenceError(f"Engine not found: {self.engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream = cuda_stream_create()

        for i in range(self._engine.num_bindings):
            dtype = self._engine.get_binding_dtype(i)
            shape = self._engine.get_binding_shape(i)
            size = trt.volume(shape)
            if self._engine.binding_is_input(i):
                self._input_shape = shape
                self._d_input = cuda_mem_alloc(size * dtype.itemsize)
            else:
                d = cuda_mem_alloc(size * dtype.itemsize)
                h = np.empty(size, dtype=trt.nptype(dtype))
                self._d_outputs.append(d)
                self._h_outputs.append(h)

        # Warmup
        dummy = np.zeros((self._input_shape[2], self._input_shape[3], 3), dtype=np.uint8)
        try:
            self.detect(dummy)
        except Exception:
            pass
        return True

    def detect(self, frame):
        h, w = self._input_shape[2], self._input_shape[3]
        blob, scale, pad = self._preprocess(frame, w, h)

        cuda_memcpy_htod(self._d_input, blob)
        bindings = [int(self._d_input)] + [int(d) for d in self._d_outputs]
        self._context.execute_async_v2(bindings, self._stream, None)
        for h_out, d_out in zip(self._h_outputs, self._d_outputs):
            cuda_memcpy_dtoh(h_out, d_out)
        cuda_stream_synchronize(self._stream)

        detections = self._postprocess(scale, pad, frame.shape[1], frame.shape[0])

        self._frame_count += 1
        if time.time() - self._fps_timer >= 1.0:
            self._fps = self._frame_count / (time.time() - self._fps_timer)
            self._frame_count = 0
            self._fps_timer = time.time()

        return detections

    def _preprocess(self, frame, tw, th):
        """Letterbox resize to target size."""
        import cv2
        h, w = frame.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        px, py = (tw - nw) // 2, (th - nh) // 2
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[py:py + nh, px:px + nw] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return blob.ravel(), scale, (px, py)

    def _postprocess(self, scale, pad, ow, oh):
        """Parse TRT output — handles [1,N,6] (NMS-processed) or raw."""
        out = self._h_outputs[0]
        total = out.size

        # Case 1a: Raw [1, 15, 8400] — YOLOv8 format (channel-major)
        if total == 126000:
            out = self._h_outputs[0].reshape(15, 8400).T  # → (8400, 15)
            return self._parse_raw_yolov8(out, scale, pad, ow, oh)

        # Case 2: Raw [1, 8400, 84] — COCO model: cx,cy,w,h + 80 class scores
        if total == 705600:
            out = out.reshape(1, 8400, 84)[0]
            return self._parse_raw(out, scale, pad, ow, oh)

        # Case 3: NMS-processed [1, N, 6] — x1,y1,x2,y2,conf,cls
        if total % self.OUTPUT_ITEMS == 0:
            n = total // self.OUTPUT_ITEMS
            if 5 <= n <= 10000:
                out = out.reshape(n, self.OUTPUT_ITEMS)
                return self._parse_processed(out, scale, pad, ow, oh)

        return []

    def _parse_processed(self, data, scale, pad, ow, oh):
        """Parse [N,6] = [x1,y1,x2,y2,conf,cls] — already NMS-filtered."""
        dets = []
        px, py = pad
        for p in data:
            if int(p[5]) != self.BALL_CLASS_ID:
                continue
            conf = float(p[4])
            if conf < self.conf_threshold:
                continue
            x1 = max(0, min(ow, (float(p[0]) - px) / scale))
            y1 = max(0, min(oh, (float(p[1]) - py) / scale))
            x2 = max(0, min(ow, (float(p[2]) - px) / scale))
            y2 = max(0, min(oh, (float(p[3]) - py) / scale))
            dets.append(Detection([x1, y1, x2, y2], conf, self.BALL_CLASS_ID))
        return dets

    def _parse_raw(self, data, scale, pad, ow, oh):
        """Parse raw detection output.

        For custom model: bbox values are already decoded [x1,y1,x2,y2].
        No padding subtraction needed — coordinates are in canvas space.
        """
        dets = []
        for p in data:
            conf = float(p[4 + self.BALL_CLASS_ID])
            if conf < self.conf_threshold:
                continue

            v0, v1, v2, v3 = float(p[0]), float(p[1]), float(p[2]), float(p[3])

            # Coordinates are in 640x640 padded canvas space
            # Adjust: remove padding, apply scale
            x1 = (v0 - pad[0]) / scale
            y1 = (v1 - pad[1]) / scale
            x2 = (v2 - pad[0]) / scale
            y2 = (v3 - pad[1]) / scale

            # Apply configurable offset/scale (fix training-data misalignment)
            cx = (x1 + x2) / 2 + self.bbox_offset_x
            cy = (y1 + y2) / 2 + self.bbox_offset_y
            bw = (x2 - x1) * self.bbox_scale_x
            bh = (y2 - y1) * self.bbox_scale_y
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            x1 = max(0, min(ow, x1))
            y1 = max(0, min(oh, y1))
            x2 = max(0, min(ow, x2))
            y2 = max(0, min(oh, y2))

            dets.append(Detection([x1, y1, x2, y2], conf, int(self.BALL_CLASS_ID)))
        return self._nms(dets)

    def _parse_raw_yolov8(self, data, scale, pad, ow, oh):
        """Parse YOLOv8 format [N, 15] = [cx,cy,w,h] + 11 class scores.
        Values in grid space with DFL decoded. Class scores may be logits.
        """
        dets = []
        for p in data:
            raw_conf = float(p[4 + self.BALL_CLASS_ID])
            # Apply sigmoid if values > 1 (TRT may fuse it)
            conf = 1.0 / (1.0 + np.exp(-raw_conf)) if raw_conf > 1 else raw_conf
            if conf < self.conf_threshold:
                continue
            cx, cy, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            x1 = (cx - w/2 - pad[0]) / scale
            y1 = (cy - h/2 - pad[1]) / scale
            x2 = (cx + w/2 - pad[0]) / scale
            y2 = (cy + h/2 - pad[1]) / scale
            bcx = (x1 + x2) / 2 + self.bbox_offset_x
            bcy = (y1 + y2) / 2 + self.bbox_offset_y
            bw = (x2 - x1) * self.bbox_scale_x
            bh = (y2 - y1) * self.bbox_scale_y
            x1 = max(0, min(ow, bcx - bw / 2))
            y1 = max(0, min(oh, bcy - bh / 2))
            x2 = max(0, min(ow, bcx + bw / 2))
            y2 = max(0, min(oh, bcy + bh / 2))
            dets.append(Detection([x1, y1, x2, y2], conf, int(self.BALL_CLASS_ID)))
        return self._nms(dets)

    def _nms(self, dets):
        if len(dets) <= 1:
            return dets
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep = []
        while dets:
            b = dets.pop(0)
            keep.append(b)
            dets = [d for d in dets if self._iou(b.bbox, d.bbox) < 0.45]
        return keep

    @staticmethod
    def _iou(a, b):
        x = max(a[0], b[0]); y = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x) * max(0, y2 - y)
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / u if u > 0 else 0.0

    @property
    def fps(self):
        return self._fps

    @property
    def is_loaded(self):
        return self._context is not None

    def unload(self):
        self._context = None
        self._engine = None
        if self._stream:
            cuda_stream_synchronize(self._stream)
        self._d_input = None
        self._d_outputs = []
        self._h_outputs = []
        import gc; gc.collect()

    def get_class_name(self, class_id):
        return self.class_names.get(class_id, f"class_{class_id}")
