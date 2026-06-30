"""
camera.py — Camera capture thread with thread-safe frame queue.

Runs a dedicated thread that reads from the C920 via OpenCV.
Provides the latest frame without blocking the main pipeline.
Supports MJPEG mode for lower USB bandwidth on Jetson Nano.
"""

import threading
import time
import cv2
import numpy as np


class CameraError(Exception):
    """Camera initialization or runtime error."""
    pass


class Camera:
    """Threaded camera capture with frame buffer."""

    def __init__(self, config):
        self.config = config
        self._cam = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self._thread = None
        self._fps = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()

    def open(self):
        """Open camera device."""
        cam_id = self.config.camera_id
        w = self.config.camera_width
        h = self.config.camera_height
        fps = self.config.camera_fps

        # OpenCV 3.2 on JP4.6 doesn't support backend arg
        try:
            self._cam = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        except TypeError:
            self._cam = cv2.VideoCapture(cam_id)
        if not self._cam.isOpened():
            raise CameraError(f"Failed to open camera {cam_id}")

        # Set MJPEG mode for lower USB bandwidth
        # On Jetson, MJPEG is critical to avoid USB bandwidth exhaustion
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        self._cam.set(cv2.CAP_PROP_FOURCC, codec)

        # Set resolution
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cam.set(cv2.CAP_PROP_FPS, fps)

        # Verify actual resolution
        actual_w = int(self._cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cam.get(cv2.CAP_PROP_FPS)

        # Drop a few frames to let auto-exposure settle
        for _ in range(10):
            self._cam.read()

        return actual_w, actual_h, actual_fps

    def start(self):
        """Start capture thread."""
        if self._running:
            return
        if self._cam is None:
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cam:
            self._cam.release()
            self._cam = None

    def read(self):
        """Get the latest frame (non-blocking, returns latest).

        Returns:
            (success, frame, frame_id) where success is False if no frames yet.
        """
        with self._lock:
            if self._frame is None:
                return False, None, 0
            return True, self._frame.copy(), self._frame_id

    @property
    def fps(self):
        return self._fps

    def _capture_loop(self):
        """Internal capture loop running in its own thread."""
        while self._running:
            if self._cam is None:
                time.sleep(0.1)
                continue

            ret, frame = self._cam.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1

            # FPS counter
            self._fps_counter += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = time.time()

    @property
    def is_running(self):
        return self._running and self._thread and self._thread.is_alive()

    def __del__(self):
        self.stop()


def list_cameras(max_id=4):
    """Probe for available cameras. Returns list of (id, name) tuples."""
    available = []
    for i in range(max_id):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, f"Camera {i} ({w}x{h})"))
            cap.release()
    return available
