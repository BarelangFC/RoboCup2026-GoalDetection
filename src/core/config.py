"""
config.py — Configuration management for Goal Detector

Loads/saves JSON config from ~/.goal-detector/config.json.
Provides typed access to all tunable parameters.
"""

import json
import os

CONFIG_DIR = os.path.expanduser("~/goal-detector/config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
TOUCH_CALIB_PATH = os.path.join(CONFIG_DIR, "touch-calib.json")
GOAL_POLYGON_PATH = os.path.join(CONFIG_DIR, "goal-polygon.json")

DEFAULT_CONFIG = {
    # --- Model ---
    "model_path": os.path.expanduser("~/yolo26n.engine"),
    "model_input_size": 640,
    "confidence_threshold": 0.45,
    "nms_threshold": 0.45,

    # --- Camera ---
    "camera_id": 0,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    "camera_backend": "cv2",  # cv2, gstreamer

    # --- Display ---
    "display_width": 800,
    "display_height": 480,
    "display_fullscreen": True,
    "show_fps": True,
    "show_debug": False,

    # --- Network (UDP Goal Event) ---
    "udp_target_ip": "192.168.123.255",
    "udp_target_port": 5000,
    "udp_bind_port": 0,
    "udp_broadcast": True,

    # --- Goal Geometry ---
    "goal_polygon": None,  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] or null

    # --- Performance ---
    "inference_threads": 1,
    "pipeline_mode": "double_buffer",  # single, double_buffer

    # --- Dataset Collection ---
    "dataset_dir": os.path.expanduser("~/goal-dataset"),
    "save_format": "jpg",
    "save_quality": 95,
}


def ensure_config_dir():
    """Create config directory if it doesn't exist."""
    os.makedirs(CONFIG_DIR, exist_ok=True)


class Config:
    """Typed configuration container."""

    def __init__(self, data=None):
        d = data if data else DEFAULT_CONFIG
        self.model_path = d.get("model_path")
        self.model_input_size = d.get("model_input_size", 640)
        self.confidence_threshold = d.get("confidence_threshold", 0.45)
        self.nms_threshold = d.get("nms_threshold", 0.45)
        self.goal_check_method = d.get("goal_check_method", "center")
        self.goal_bbox_overlap_pct = d.get("goal_bbox_overlap_pct", 0.5)
        self.bbox_offset_x = d.get("bbox_offset_x", 0)
        self.bbox_offset_y = d.get("bbox_offset_y", 0)
        self.bbox_scale_x = d.get("bbox_scale_x", 1.0)
        self.bbox_scale_y = d.get("bbox_scale_y", 1.0)

        self.camera_id = d.get("camera_id", 0)
        self.camera_width = d.get("camera_width", 640)
        self.camera_height = d.get("camera_height", 480)
        self.camera_fps = d.get("camera_fps", 30)
        self.camera_backend = d.get("camera_backend", "cv2")

        self.display_width = d.get("display_width", 800)
        self.display_height = d.get("display_height", 480)
        self.display_fullscreen = d.get("display_fullscreen", True)
        self.show_fps = d.get("show_fps", True)
        self.show_debug = d.get("show_debug", False)

        self.udp_target_ip = d.get("udp_target_ip", "192.168.123.255")
        self.udp_target_port = d.get("udp_target_port", 5000)
        self.udp_bind_port = d.get("udp_bind_port", 0)
        self.udp_broadcast = d.get("udp_broadcast", True)

        self.goal_polygon = d.get("goal_polygon", None)

        self.inference_threads = d.get("inference_threads", 1)
        self.pipeline_mode = d.get("pipeline_mode", "double_buffer")

        self.dataset_dir = d.get("dataset_dir", os.path.expanduser("~/goal-dataset"))
        self.save_format = d.get("save_format", "jpg")
        self.save_quality = d.get("save_quality", 95)

    def to_dict(self):
        return vars(self)

    def save(self):
        """Save current config to disk."""
        ensure_config_dir()
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load():
        """Load config from disk, merging with defaults for any missing keys."""
        if not os.path.exists(CONFIG_PATH):
            cfg = Config()
            cfg.save()
            return cfg
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return Config(merged)


def save_touch_calibration(matrix):
    """Save 3x3 touch calibration matrix."""
    ensure_config_dir()
    with open(TOUCH_CALIB_PATH, "w") as f:
        json.dump({"matrix": matrix}, f, indent=2)


def load_touch_calibration():
    """Load 3x3 touch calibration matrix, or return default identity."""
    if not os.path.exists(TOUCH_CALIB_PATH):
        return None
    try:
        with open(TOUCH_CALIB_PATH, "r") as f:
            return json.load(f).get("matrix")
    except (json.JSONDecodeError, IOError):
        return None


def save_goal_polygon(polygon):
    """Save goal polygon as list of 4 [x,y] points."""
    ensure_config_dir()
    with open(GOAL_POLYGON_PATH, "w") as f:
        json.dump({"polygon": polygon, "version": 1}, f, indent=2)


def load_goal_polygon():
    """Load goal polygon, or return None."""
    if not os.path.exists(GOAL_POLYGON_PATH):
        return None
    try:
        with open(GOAL_POLYGON_PATH, "r") as f:
            return json.load(f).get("polygon")
    except (json.JSONDecodeError, IOError):
        return None


def set_udp_target(ip, port):
    """Quick helper to update UDP target without full config reload."""
    cfg = Config.load()
    cfg.udp_target_ip = ip
    cfg.udp_target_port = port
    cfg.save()
    return cfg
