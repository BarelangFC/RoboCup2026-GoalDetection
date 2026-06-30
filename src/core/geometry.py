"""
geometry.py — Goal geometry checking with multiple overlap modes.

Modes:
  "center"     — ball bottom-center point must be inside goal polygon
  "full"       — entire ball bbox must be inside goal polygon
  "overlap_pct" — configurable % of bbox area must overlap goal polygon
"""

import numpy as np
import cv2


class GoalChecker:
    """Checks if a ball has entered the goal area."""

    def __init__(self):
        self._polygon = None      # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        self._poly_np = None      # numpy array
        self._goal_scored = False
        self._last_goal_time = 0.0
        self._cooldown = 1.0
        self.check_method = "center"
        self.overlap_pct = 0.5

    def set_polygon(self, polygon):
        if polygon is None or len(polygon) < 3:
            self._polygon = None
            self._poly_np = None
            return
        self._polygon = polygon
        self._poly_np = np.array(polygon, dtype=np.int32)

    def get_polygon(self):
        return self._polygon

    def clear_polygon(self):
        self._polygon = None
        self._poly_np = None

    @property
    def has_polygon(self):
        return self._polygon is not None and len(self._polygon) >= 3

    def check_ball(self, bbox, current_time):
        """Check if ball detection indicates a goal.

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            current_time: time.time()

        Returns:
            (is_goal, test_point)
        """
        if not self.has_polygon:
            return False, None

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, y2  # bottom-center (ground contact)

        result = False
        point = None

        if self.check_method == "center":
            # Ball bottom-center point inside polygon
            inside = self._point_in_poly(cx, cy)
            result = inside
            point = (int(cx), int(cy))

        elif self.check_method == "full":
            # Entire bbox must be inside polygon
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            all_inside = all(self._point_in_poly(px, py) for px, py in corners)
            result = all_inside
            point = (int(cx), int(cy))

        elif self.check_method == "overlap_pct":
            # Calculate bbox-goal overlap percentage
            overlap = self._bbox_goal_overlap(x1, y1, x2, y2)
            result = overlap >= self.overlap_pct
            point = (int(cx), int(cy))

        if result:
            if current_time - self._last_goal_time > self._cooldown:
                self._goal_scored = True
                self._last_goal_time = current_time
                return True, point
            else:
                return False, point

        return False, point

    def _point_in_poly(self, px, py):
        """Point-in-polygon test via OpenCV."""
        if self._poly_np is None:
            return False
        contour = self._poly_np.reshape((-1, 1, 2)).astype(np.int32)
        return cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0

    def _bbox_goal_overlap(self, x1, y1, x2, y2):
        """Calculate overlap percentage between bbox and goal polygon.

        Uses pixel-level sampling for speed on embedded devices.
        """
        if self._poly_np is None or x2 <= x1 or y2 <= y1:
            return 0.0

        # Create a mask of the goal polygon
        bw, bh = int(x2 - x1), int(y2 - y1)
        if bw <= 0 or bh <= 0:
            return 0.0

        # Sample grid for speed (max 20x20 samples)
        step_x = max(1, bw // 20)
        step_y = max(1, bh // 20)
        total = 0
        inside = 0

        for py in np.arange(y1, y2, step_y):
            for px in np.arange(x1, x2, step_x):
                total += 1
                if self._point_in_poly(px, py):
                    inside += 1

        return inside / total if total > 0 else 0.0

    def reset_goal(self):
        self._goal_scored = False

    @property
    def goal_just_scored(self):
        return self._goal_scored

    def get_polygon_for_drawing(self):
        if self._poly_np is None:
            return None
        return self._poly_np.reshape((-1, 2))
