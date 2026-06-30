#!/usr/bin/env python3
"""
main.py — Goal Detector State Machine & Main Loop

Jetson Nano 2GB | RoboCup Goal Detection
TensorRT YOLO26n | Pygame KMS/DRM | UDP Event Dispatch

States:
    BOOT              → Load model, init camera, load config
    MAIN_MENU         → Touch button selection
    TEST_MODE         → Live feed + overlay + FPS
    RUN_MODE          → Production goal detection + UDP dispatch
    DATASET_COLLECT   → Tap screen to save frames to disk
    CALIBRATION       → 4-point tap to define scoring polygon
"""

import os
import sys
import time
import signal
import threading
import json
import argparse
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ─── CLI Arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Goal Detector")
parser.add_argument("--display", help="X11 display (e.g. :1 for NoMachine)")
parser.add_argument("--headless", action="store_true", help="Run without GUI (console only)")
parser.add_argument("--fullscreen", action="store_true", help="Force fullscreen")
parser.add_argument("--no-fullscreen", action="store_false", dest="fullscreen")
parser.add_argument("--size", help="Window size WxH (e.g. 1024x600)")
args, _ = parser.parse_known_args()

# Configure video backend based on CLI args
if args.headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
elif args.display:
    os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ["DISPLAY"] = args.display
# else: SDL auto-detects (HDMI, KMS/DRM, etc.)

import pygame

from core.config import Config, save_goal_polygon, load_goal_polygon
from core.camera import Camera, CameraError
from core.geometry import GoalChecker
from core.network import UdpDispatcher

# ─── Detector Selection ───────────────────────────────────────────
# Prefer pure TensorRT backend (no ultralytics dependency at runtime).
# Fall back to ultralytics-based inference if trt is unavailable.
Detector = None
InferenceError = None
try:
    from core.trt_inference import TRTYoloDetector as Detector, TRTInferenceError as InferenceError
    print("[BOOT] Using pure TensorRT inference backend")
except ImportError:
    try:
        from core.inference import YoloDetector as Detector, InferenceError
        print("[BOOT] Using Ultralytics inference backend")
    except ImportError:
        pass

from ui_components import (
    Button, FpsCounter, OverlayRenderer, COLORS,
)

# ─── State Constants ──────────────────────────────────────────────
STATE_BOOT = "BOOT"
STATE_MAIN_MENU = "MAIN_MENU"
STATE_TEST_MODE = "TEST_MODE"
STATE_RUN_MODE = "RUN_MODE"
STATE_DATASET_COLLECT = "DATASET_COLLECT"
STATE_CALIBRATION = "CALIBRATION"


class GoalDetectorApp:
    """Main application state machine."""

    def __init__(self):
        self.config = Config.load()
        self.cli_args = args
        self.state = STATE_BOOT
        self.next_state = None
        self.running = True

        # Subsystems
        self.camera = None
        self.detector = None
        self.goal_checker = GoalChecker()
        self.network = UdpDispatcher(self.config)
        self.overlay = OverlayRenderer()
        self.fps_counter = FpsCounter(10, 10)

        # Display
        self.screen = None
        self.clock = None
        self.display_size = (self.config.display_width,
                             self.config.display_height)
        self.camera_frame_size = (self.config.camera_width,
                                  self.config.camera_height)

        # State-specific
        self.menu_buttons = []
        self.dataset_count = 0
        self.calibration_step = 0
        self.calibration_points = []
        self.goal_flash_timer = 0.0
        self.frame_buffer = None
        self.last_detections = []
        self.last_inference_time = 0.0
        self.goal_events_dispatched = 0
        self.ball_detected = False

        # Timing
        self.frame_interval = 1.0 / 30
        self.last_frame_time = 0.0
        self.start_time = time.time()

    # ─── Boot / Init ──────────────────────────────────────────────

    def _draw_loading(self, text, progress=0.0):
        """Draw a loading screen with optional progress bar."""
        if self.screen is None:
            return
        self.screen.fill(COLORS["bg_dark"])
        font = pygame.font.Font(None, 36)
        title = font.render("GOAL DETECTOR", True, COLORS["text_primary"])
        tr = title.get_rect(center=(self.display_size[0] // 2, self.display_size[1] // 2 - 60))
        self.screen.blit(title, tr)

        status = pygame.font.Font(None, 28).render(text, True, COLORS["text_secondary"])
        sr = status.get_rect(center=(self.display_size[0] // 2, self.display_size[1] // 2))
        self.screen.blit(status, sr)

        if progress > 0:
            bar_w, bar_h = 400, 20
            bx = (self.display_size[0] - bar_w) // 2
            by = self.display_size[1] // 2 + 30
            pygame.draw.rect(self.screen, COLORS["button_normal"], (bx, by, bar_w, bar_h), 2)
            fill_w = int(bar_w * min(1.0, progress))
            if fill_w > 0:
                pygame.draw.rect(self.screen, COLORS["accent_blue"], (bx + 2, by + 2, fill_w - 4, bar_h - 4))

        pygame.display.flip()

    def boot(self):
        """Initialize all subsystems."""
        print("[BOOT] Loading configuration...")
        # ... existing boot code ...

        # Start inference thread
        self._inference_thread = None
        self._inference_lock = threading.Lock()
        self._inference_running = False
        self._latest_frame_for_inf = None
        print(f"       Model: {self.config.model_path}")
        print(f"       Display: {self.display_size[0]}x{self.display_size[1]}")
        print(f"       Camera: {self.config.camera_width}x{self.config.camera_height}")
        print(f"       UDP: {self.config.udp_target_ip}:{self.config.udp_target_port}")

        # Load goal polygon
        polygon = load_goal_polygon()
        if polygon:
            self.goal_checker.set_polygon(polygon)
            print(f"[BOOT] Goal polygon loaded: {len(polygon)} points")
        self.goal_checker.check_method = self.config.goal_check_method
        self.goal_checker.overlap_pct = self.config.goal_bbox_overlap_pct

        # Init display FIRST so we can show loading status
        self._init_display()
        self._draw_loading("Initializing subsystems...", 0.1)

        # Init network (lightweight, do early)
        self.network.open()
        print(f"[BOOT] UDP dispatcher ready")

        # Load model BEFORE camera (model needs GPU memory; camera buffers eat RAM)
        self._draw_loading("Loading TensorRT engine...", 0.3)
        self._init_detector()
        import gc; gc.collect()

        # Init camera AFTER model (camera frame buffers are ~3 MB, no GPU needed)
        self._draw_loading("Opening camera...", 0.7)
        self._init_camera()

        # Final setup
        self._draw_loading("Ready!", 1.0)
        self.state = STATE_MAIN_MENU
        self._setup_main_menu()
        print("[BOOT] System ready — entering MAIN_MENU")

    def _detect_display_size(self):
        """Detect actual display size via xrandr or X11."""
        display = self.cli_args.display or os.environ.get("DISPLAY", ":0")
        try:
            import subprocess
            r = subprocess.run(
                ["xrandr", "--display", display, "--current"],
                capture_output=True, text=True, timeout=5
            )
            for line in r.stdout.split("\n"):
                if " connected" in line or "Screen" in line:
                    # "Screen 0: minimum 8 x 8, current 640 x 480, ..."
                    # "HDMI-0 connected primary 1920x1080+0+0 ..."
                    import re
                    m = re.search(r"current (\d+) x (\d+)|connected.*?(\d+)x(\d+)", line)
                    if m:
                        groups = m.groups()
                        w = int(groups[0] or groups[2])
                        h = int(groups[1] or groups[3])
                        if w > 0 and h > 0:
                            return (w, h)
        except Exception:
            pass
        return None

    def _init_display(self):
        """Initialise Pygame display with auto-detected size."""
        # Auto-detect display size via xrandr (before pygame init)
        detected = self._detect_display_size()
        if detected:
            print(f"[BOOT] Detected display: {detected[0]}x{detected[1]}")
            self.display_size = detected

        # Override with --size if provided
        if self.cli_args.size:
            try:
                w, h = self.cli_args.size.lower().split("x")
                self.display_size = (int(w), int(h))
                print(f"[BOOT] Using --size: {self.display_size[0]}x{self.display_size[1]}")
            except Exception:
                print(f"[WARN] Invalid --size: {self.cli_args.size}, using detected")

        pygame.init()
        use_fullscreen = self.cli_args.fullscreen if self.cli_args.fullscreen is not None else self.config.display_fullscreen

        if self.cli_args.headless:
            print("[BOOT] Headless mode -- creating dummy display")
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.display.set_mode((1, 1))
            return

        flags = pygame.FULLSCREEN | pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
        if use_fullscreen:
            self.screen = pygame.display.set_mode(self.display_size, flags)
        else:
            self.screen = pygame.display.set_mode(self.display_size)
        pygame.display.set_caption("Goal Detector")
        self.clock = pygame.time.Clock()
        # Show cursor for X11/NoMachine; hide on physical touch display
        is_x11 = pygame.display.get_driver() == "x11"
        pygame.mouse.set_visible(is_x11)
        print(f"[BOOT] Display: {self.display_size[0]}x{self.display_size[1]} ({pygame.display.get_driver()})")

    def _init_camera(self):
        """Initialize and start camera."""
        print("[BOOT] Opening camera...")
        self.camera = Camera(self.config)
        try:
            actual_w, actual_h, actual_fps = self.camera.open()
            print(f"       Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
            self.camera_frame_size = (actual_w, actual_h)
            self.camera.start()
        except CameraError as e:
            print(f"[ERROR] Camera: {e}")
            self.camera = None

    def _init_detector(self):
        """Load TensorRT model."""
        print(f"[BOOT] Loading model: {self.config.model_path}")
        self.detector = Detector(
            self.config.model_path,
            conf_threshold=self.config.confidence_threshold,
            nms_threshold=self.config.nms_threshold,
        )
        self.detector.bbox_offset_x = self.config.bbox_offset_x
        self.detector.bbox_offset_y = self.config.bbox_offset_y
        self.detector.bbox_scale_x = self.config.bbox_scale_x
        self.detector.bbox_scale_y = self.config.bbox_scale_y
        try:
            self.detector.load()
            print(f"[BOOT] Model loaded successfully")
        except InferenceError as e:
            print(f"[ERROR] Model: {e}")
            self.detector = None

    # ─── Main Menu ─────────────────────────────────────────────────

    def _setup_main_menu(self):
        """Create main menu touch buttons."""
        dw, dh = self.display_size
        bw, bh = 280, 80
        gap = 20
        start_y = dh // 2 - (bh * 4 + gap * 3) // 2
        cx = dw // 2

        def make_cb(target_state):
            return lambda: self._transition(target_state)

        self.menu_buttons = [
            Button((cx - bw // 2, start_y + 0 * (bh + gap), bw, bh),
                   "TEST MODE", make_cb(STATE_TEST_MODE),
                   color=COLORS["accent_blue"]),

            Button((cx - bw // 2, start_y + 1 * (bh + gap), bw, bh),
                   "RUN MODE", make_cb(STATE_RUN_MODE),
                   color=COLORS["accent_green"]),

            Button((cx - bw // 2, start_y + 2 * (bh + gap), bw, bh),
                   "DATASET COLLECT", make_cb(STATE_DATASET_COLLECT),
                   color=COLORS["accent_yellow"]),

            Button((cx - bw // 2, start_y + 3 * (bh + gap), bw, bh),
                   "CALIBRATION", make_cb(STATE_CALIBRATION),
                   color=COLORS["accent_red"]),
        ]

    def _render_main_menu(self):
        """Draw main menu screen."""
        self.screen.fill(COLORS["bg_dark"])

        # Title
        font = pygame.font.Font(None, 48)
        title = font.render("GOAL DETECTOR", True, COLORS["text_primary"])
        title_rect = title.get_rect(center=(self.display_size[0] // 2, 60))
        self.screen.blit(title, title_rect)

        # Subtitle
        font2 = pygame.font.Font(None, 20)
        sub = font2.render("Tap a mode to begin", True, COLORS["text_secondary"])
        sub_rect = sub.get_rect(center=(self.display_size[0] // 2, 95))
        self.screen.blit(sub, sub_rect)

        # Status indicators
        model_status = "MODEL: READY" if (self.detector and self.detector.is_loaded) else "MODEL: NOT LOADED"
        cam_status = "CAM: OK" if (self.camera and self.camera.is_running) else "CAM: OFF"
        goal_status = "GOAL: CALIBRATED" if self.goal_checker.has_polygon else "GOAL: NOT SET"

        font3 = pygame.font.Font(None, 18)
        for i, txt in enumerate([model_status, cam_status, goal_status]):
            color = COLORS["accent_green"] if "READY" in txt or "OK" in txt or "CALIBRATED" in txt else COLORS["accent_red"]
            surf = font3.render(txt, True, color)
            self.screen.blit(surf, (20, self.display_size[1] - 60 + i * 20))

        # Buttons
        for btn in self.menu_buttons:
            btn.draw(self.screen)

    # ─── Test Mode ─────────────────────────────────────────────────

    def _enter_test_mode(self):
        self.goal_flash_timer = 0.0
        self.goal_checker.reset_goal()

    def _render_test_mode(self):
        """Live feed + detection overlay + FPS + goal check."""
        self.screen.fill(COLORS["bg_dark"])

        # Get latest frame
        ret, frame, fid = self.camera.read() if self.camera else (False, None, 0)

        goal_now = False
        if ret and frame is not None:
            # Run inference
            if self.detector and self.detector.is_loaded:
                self.last_detections = self.detector.detect(frame)
                inference_fps = self.detector.fps

                # Geometry check for each ball detection (goal test)
                now = time.time()
                for det in self.last_detections:
                    is_goal, point = self.goal_checker.check_ball(det.bbox, now)
                    if is_goal:
                        goal_now = True
                        self.goal_flash_timer = now + 1.0
                        print(f"[TEST] GOAL! Ball at {point}")
            else:
                self.last_detections = []
                inference_fps = 0.0

            # Scale frame to fit display
            frame_h, frame_w = frame.shape[:2]
            scale = min(self.display_size[0] / frame_w,
                        self.display_size[1] / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            disp_frame = cv2_resize(frame, (new_w, new_h))

            # Convert OpenCV BGR to RGB for Pygame
            disp_frame = cv2_cvt_color(disp_frame, cv2_color_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(disp_frame.swapaxes(0, 1))
            offset_x = (self.display_size[0] - new_w) // 2
            offset_y = (self.display_size[1] - new_h) // 2
            self.screen.blit(frame_surf, (offset_x, offset_y))

            # Draw detection overlay on camera frame area
            self.overlay.draw_goal_polygon(self.screen,
                                           self.goal_checker.get_polygon_for_drawing())
            self.overlay.draw_detections(self.screen, self.last_detections,
                                         self.detector)

        # Goal flash overlay
        if goal_now or time.time() < self.goal_flash_timer:
            self.overlay.draw_goal_flash(self.screen)

        # FPS counter
        cam_fps = self.camera.fps if self.camera else 0.0
        inf_fps = self.detector.fps if self.detector else 0.0
        self.fps_counter.draw(self.screen, cam_fps,
                              f"det: {inf_fps:.1f} | {len(self.last_detections)} balls")

        # Goal status indicator
        goal_status = "GOAL: READY" if self.goal_checker.has_polygon else "NO GOAL POLYGON"
        goal_color = COLORS["accent_green"] if self.goal_checker.has_polygon else COLORS["accent_red"]
        gs = pygame.font.Font(None, 22).render(goal_status, True, goal_color)
        self.screen.blit(gs, (self.display_size[0] - 200, self.display_size[1] - 30))

        # Back button
        self._draw_back_button()

    # ─── Run Mode ──────────────────────────────────────────────────

    def _enter_run_mode(self):
        """Enter production mode — minimal UI, max compute."""
        if not self.goal_checker.has_polygon:
            print("[RUN] WARNING: No goal polygon calibrated!")
        self.goal_events_dispatched = 0
        self.goal_checker.reset_goal()
        self.goal_flash_timer = 0.0

    def _render_run_mode(self):
        """Production mode — event dispatch focus."""
        self.screen.fill(COLORS["bg_dark"])

        # Get frame
        ret, frame, fid = self.camera.read() if self.camera else (False, None, 0)

        goal_event = False
        if ret and frame is not None:
            # Inference
            if self.detector and self.detector.is_loaded:
                detections = self.detector.detect(frame)
                self.last_detections = detections

                # Geometry check for each ball detection
                now = time.time()
                for det in detections:
                    is_goal, point = self.goal_checker.check_ball(det.bbox, now)
                    if is_goal:
                        goal_event = True
                        self.network.send_goal(confidence=det.confidence)
                        self.goal_events_dispatched += 1
                        self.goal_flash_timer = now + 1.0
                        print(f"[GOAL] Goal detected! Event #{self.goal_events_dispatched}")

                # Send heartbeat if no ball detected
                if not detections:
                    if self.goal_checker.goal_just_scored:
                        pass  # Don't override goal state
                    self.ball_detected = False

            # Show minimized frame to save GPU memory for inference
            if ret and frame is not None and self.config.show_debug:
                frame_h, frame_w = frame.shape[:2]
                scale = min(self.display_size[0] / frame_w,
                            self.display_size[1] / frame_h) * 0.4
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)
                disp_frame = cv2_resize(frame, (new_w, new_h))
                disp_frame = cv2_cvt_color(disp_frame, cv2_color_BGR2RGB)
                frame_surf = pygame.surfarray.make_surface(disp_frame.swapaxes(0, 1))
                offset_x = (self.display_size[0] - new_w) // 2
                offset_y = (self.display_size[1] - new_h) // 2
                self.screen.blit(frame_surf, (offset_x, offset_y))
                self.overlay.draw_goal_polygon(self.screen,
                                               self.goal_checker.get_polygon_for_drawing())

        # Goal flash overlay
        if goal_event or time.time() < self.goal_flash_timer:
            self.overlay.draw_goal_flash(self.screen)

        # Status UI
        font = pygame.font.Font(None, 36)
        status = f"RUN MODE — Goals: {self.goal_events_dispatched}"
        surf = font.render(status, True, COLORS["accent_green"])
        self.screen.blit(surf, (20, 20))

        if not self.goal_checker.has_polygon:
            warn = pygame.font.Font(None, 28).render(
                "NO GOAL POLYGON — Calibrate first!", True, COLORS["accent_red"])
            self.screen.blit(warn, (20, 60))

        self._draw_back_button()

    # ─── Dataset Collection Mode ──────────────────────────────────

    def _enter_dataset_collect(self):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        self.dataset_count = len([
            f for f in os.listdir(self.config.dataset_dir)
            if f.endswith("." + self.config.save_format)
        ])
        print(f"[DATASET] Collecting to {self.config.dataset_dir}")
        print(f"[DATASET] Existing images: {self.dataset_count}")

    def _save_frame(self):
        """Save current camera frame to dataset directory."""
        ret, frame, fid = self.camera.read() if self.camera else (False, None, 0)
        if not ret or frame is None:
            return

        self.dataset_count += 1
        ext = self.config.save_format
        fname = f"frame_{self.dataset_count:06d}.{ext}"
        fpath = os.path.join(self.config.dataset_dir, fname)

        import cv2
        if ext == "jpg":
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.save_quality])
        else:
            cv2.imwrite(fpath, frame)
        print(f"[DATASET] Saved: {fpath}")

    def _render_dataset_collect(self):
        """Show live feed with tap-to-save."""
        self.screen.fill(COLORS["bg_dark"])

        ret, frame, fid = self.camera.read() if self.camera else (False, None, 0)
        if ret and frame is not None:
            frame_h, frame_w = frame.shape[:2]
            scale = min(self.display_size[0] / frame_w,
                        self.display_size[1] / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            disp_frame = cv2_resize(frame, (new_w, new_h))
            disp_frame = cv2_cvt_color(disp_frame, cv2_color_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(disp_frame.swapaxes(0, 1))
            ox = (self.display_size[0] - new_w) // 2
            oy = (self.display_size[1] - new_h) // 2
            self.screen.blit(frame_surf, (ox, oy))

        # Overlay instructions
        font = pygame.font.Font(None, 32)
        info = font.render(f"TAP TO SAVE — Count: {self.dataset_count}",
                           True, COLORS["text_primary"])
        self.screen.blit(info, (20, 20))

        hint = pygame.font.Font(None, 22).render(
            "Tap anywhere to capture the current frame",
            True, COLORS["text_secondary"])
        self.screen.blit(hint, (20, 55))

        self._draw_back_button()

    # ─── Calibration Mode ─────────────────────────────────────────

    def _enter_calibration(self):
        self.calibration_step = 0
        self.calibration_points = []

    def _handle_calibration_tap(self, pos):
        """Record a calibration point on tap."""
        x, y = pos
        self.calibration_points.append([x, y])
        self.calibration_step += 1

        if self.calibration_step >= 4:
            # 4 points collected: order is TL, TR, BR, BL
            self.goal_checker.set_polygon(self.calibration_points)
            save_goal_polygon(self.calibration_points)
            print(f"[CALIB] Goal polygon saved: {self.calibration_points}")
            self._transition(STATE_MAIN_MENU)

    def _render_calibration(self):
        """Show calibration overlay — tap 4 corners of goal."""
        self.screen.fill(COLORS["bg_dark"])

        # Show camera feed
        ret, frame, fid = self.camera.read() if self.camera else (False, None, 0)
        if ret and frame is not None:
            frame_h, frame_w = frame.shape[:2]
            scale = min(self.display_size[0] / frame_w,
                        self.display_size[1] / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            disp_frame = cv2_resize(frame, (new_w, new_h))
            disp_frame = cv2_cvt_color(disp_frame, cv2_color_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(disp_frame.swapaxes(0, 1))
            ox = (self.display_size[0] - new_w) // 2
            oy = (self.display_size[1] - new_h) // 2
            self.screen.blit(frame_surf, (ox, oy))

        # Draw collected points + polygon
        if len(self.calibration_points) >= 2:
            pts = self.calibration_points
            # Draw lines between collected points
            for i in range(len(pts) - 1):
                pygame.draw.line(self.screen, COLORS["accent_yellow"],
                                 pts[i], pts[i + 1], 3)
            if len(pts) == 4:
                pygame.draw.line(self.screen, COLORS["accent_yellow"],
                                 pts[3], pts[0], 3)

        # Draw each point
        for i, pt in enumerate(self.calibration_points):
            pygame.draw.circle(self.screen, COLORS["accent_green"],
                               pt, 10)
            pygame.draw.circle(self.screen, COLORS["text_primary"],
                               pt, 10, 2)
            font = pygame.font.Font(None, 28)
            label = font.render(str(i + 1), True, COLORS["text_primary"])
            self.screen.blit(label, (pt[0] + 14, pt[1] - 10))

        # Instructions overlay
        instructions = [
            f"Step {self.calibration_step + 1}/4: Tap the corners of the goal",
            "1 → Top-Left    2 → Top-Right",
            "3 → Bottom-Right    4 → Bottom-Left",
        ]
        font = pygame.font.Font(None, 24)
        for i, txt in enumerate(instructions):
            surf = font.render(txt, True, COLORS["text_primary"])
            self.screen.blit(surf, (20, 20 + i * 28))

        self._draw_back_button()

    # ─── Helpers ──────────────────────────────────────────────────

    def _transition(self, new_state):
        """Schedule a state transition."""
        self.next_state = new_state

    def _apply_transition(self):
        """Execute state transition."""
        if self.next_state is None:
            return
        print(f"[STATE] {self.state} -> {self.next_state}")
        self.state = self.next_state
        self.next_state = None

        # Call enter handlers
        enter_map = {
            STATE_TEST_MODE: self._enter_test_mode,
            STATE_RUN_MODE: self._enter_run_mode,
            STATE_DATASET_COLLECT: self._enter_dataset_collect,
            STATE_CALIBRATION: self._enter_calibration,
        }
        handler = enter_map.get(self.state)
        if handler:
            handler()

        # Clear frame buffer on state transition to free memory
        self.last_detections = []
        self._back_button = None  # Reset back button for new state
        import gc
        gc.collect()

    def _draw_back_button(self):
        """Draw a back button in the top-right corner."""
        dw = self.display_size[0]
        if not hasattr(self, '_back_button') or self._back_button is None:
            self._back_button = Button(
                (dw - 120, 10, 110, 50), "BACK",
                lambda: self._transition(STATE_MAIN_MENU),
                color=(80, 30, 30), font_size=24
            )
        self._back_button.draw(self.screen)

    # ─── Event Loop ───────────────────────────────────────────────

    def handle_events(self):
        """Process pygame events for the current state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # Handle menu buttons
            if self.state == STATE_MAIN_MENU:
                for btn in self.menu_buttons:
                    btn.handle_event(event)

            # Handle back button (if drawn this frame)
            if hasattr(self, '_back_button') and self._back_button:
                self._back_button.handle_event(event)

            # Handle calibration taps
            if self.state == STATE_CALIBRATION:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check if not on back button
                    if not (hasattr(self, '_back_button') and
                            self._back_button and
                            self._back_button.rect.collidepoint(event.pos)):
                        self._handle_calibration_tap(event.pos)
                elif event.type == pygame.FINGERDOWN:
                    w, h = self.display_size
                    px = int(event.x * w)
                    py = int(event.y * h)
                    if not (hasattr(self, '_back_button') and
                            self._back_button and
                            self._back_button.rect.collidepoint((px, py))):
                        self._handle_calibration_tap((px, py))

            # Handle dataset collection tap
            if self.state == STATE_DATASET_COLLECT:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not (hasattr(self, '_back_button') and
                            self._back_button and
                            self._back_button.rect.collidepoint(event.pos)):
                        self._save_frame()
                elif event.type == pygame.FINGERDOWN:
                    self._save_frame()

    # ─── Main Loop ────────────────────────────────────────────────

    def run(self):
        """Main application loop."""
        # Boot
        self.boot()

        # Main loop
        while self.running:
            frame_start = time.time()

            # Handle state transition
            self._apply_transition()

            # Events
            self.handle_events()

            # Render current state
            render_map = {
                STATE_MAIN_MENU: self._render_main_menu,
                STATE_TEST_MODE: self._render_test_mode,
                STATE_RUN_MODE: self._render_run_mode,
                STATE_DATASET_COLLECT: self._render_dataset_collect,
                STATE_CALIBRATION: self._render_calibration,
            }
            render_fn = render_map.get(self.state)
            if render_fn:
                render_fn()

            # Update display (skip if headless)
            if not self.cli_args.headless:
                pygame.display.flip()

            # Frame rate limiting
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                self.clock.tick(60)
            else:
                self.clock.tick()

    def shutdown(self):
        """Clean shutdown of all subsystems."""
        print("\n[SHUTDOWN] ...")
        if self.camera:
            self.camera.stop()
        if self.network:
            self.network.close()
        if self.detector:
            self.detector.unload()
        pygame.quit()
        print("[SHUTDOWN] Complete")


# ─── OpenCV helpers (lightweight imports to avoid top-level delays) ──

def cv2_resize(frame, size):
    import cv2
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def cv2_cvt_color(frame, code):
    import cv2
    return cv2.cvtColor(frame, code)


cv2_color_BGR2RGB = 4  # cv2.COLOR_BGR2RGB


# ─── Entry Point ──────────────────────────────────────────────────

def signal_handler(sig, frame):
    print("\n[SIGNAL] Interrupt received")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = GoalDetectorApp()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.shutdown()
