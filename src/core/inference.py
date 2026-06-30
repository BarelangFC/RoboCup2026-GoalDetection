"""
inference.py — YOLO detection via TensorRT engine.

Wraps the Ultralytics YOLO model loaded from a TensorRT .engine file.
Since we use detection-only (1 class: ball), every detection result
contains bounding box + confidence for the ""sports ball"" class.

On Maxwell GPU (Nano), the engine should be exported as FP32 or INT8.
"""

import os
import time
import numpy as np
from collections import namedtuple

# Detection result data class
Detection = namedtuple("Detection", ["bbox", "confidence", "class_id"])


class InferenceError(Exception):
    pass


class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.45, nms_threshold=0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._model = None
        self._input_size = 640
        self._class_names = []
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

    def load(self):
        from ultralytics import YOLO
        if not os.path.exists(self.model_path):
            raise InferenceError(f"Model not found: {self.model_path}")
        self._model = YOLO(self.model_path)
        self._input_size = self._model.model.args.get("imgsz", 640) if hasattr(self._model, "model") else 640
        self._class_names = self._model.names if hasattr(self._model, "names") else {}
        # Warmup inference
        self._warmup()
        return True

    def _warmup(self):
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        try:
            self._model(dummy, verbose=False)
        except Exception:
            pass

    def detect(self, frame):
        if self._model is None:
            raise InferenceError("Model not loaded. Call load() first.")

        results = self._model(
            frame,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            verbose=False,
            device=0,
        )

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
                    detections.append(Detection(
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                        class_id=int(cls_id),
                    ))

        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

        return detections

    @property
    def fps(self):
        return self._fps

    @property
    def is_loaded(self):
        return self._model is not None

    def unload(self):
        self._model = None
        import gc
        gc.collect()

    def get_class_name(self, class_id):
        return self._class_names.get(class_id, f"class_{class_id}")
