"""
ui_components.py — Touch-friendly UI components rendered via Pygame.

Designed for 800x480 resistive touch display with xpt2046 controller.
All UI elements are large, high-contrast, and spaced for finger input.
"""

import pygame


# Colour palette (high contrast for outdoor/field use)
COLORS = {
    "bg_dark": (20, 20, 30),
    "bg_light": (40, 40, 55),
    "text_primary": (240, 240, 255),
    "text_secondary": (180, 180, 200),
    "accent_blue": (50, 120, 220),
    "accent_green": (50, 200, 80),
    "accent_red": (220, 50, 50),
    "accent_yellow": (220, 200, 40),
    "button_normal": (60, 60, 80),
    "button_hover": (80, 80, 110),
    "button_active": (50, 50, 200),
    "goal_polygon": (0, 255, 0, 60),     # Semi-transparent green fill
    "goal_outline": (0, 255, 0),
    "ball_bbox": (255, 200, 0),
    "ball_center": (255, 255, 0),
    "fps_text": (100, 255, 100),
    "goal_flash": (255, 0, 0),
}


class Button:
    """A touch-optimised button widget.

    Minimum size: 120x60 pixels for finger accuracy on resistive touch.
    """

    def __init__(self, rect, label, callback=None, color=None,
                 label_color=None, border_radius=8, font_size=28):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.color = color or COLORS["button_normal"]
        self.label_color = label_color or COLORS["text_primary"]
        self.border_radius = border_radius
        self.font_size = font_size
        self.hovered = False
        self.active = False
        self._font = None
        self._label_surf = None

    def _get_font(self):
        if self._font is None:
            try:
                self._font = pygame.font.Font(None, self.font_size)
            except Exception:
                self._font = pygame.font.Font(None, self.font_size)
        return self._font

    def _render_label(self):
        font = self._get_font()
        # Simple word wrap if needed
        words = self.label.split("\\n")
        lines = []
        for word in words:
            lines.append(font.render(word, True, self.label_color))
        self._label_surf = lines

    def draw(self, surface):
        if self._label_surf is None:
            self._render_label()

        # Background
        if self.active:
            bg = COLORS["button_active"]
        elif self.hovered:
            bg = COLORS["button_hover"]
        else:
            bg = self.color

        pygame.draw.rect(surface, bg, self.rect,
                         border_radius=self.border_radius)
        pygame.draw.rect(surface, COLORS["text_secondary"], self.rect, 2,
                         border_radius=self.border_radius)

        # Label centered
        total_h = sum(s.get_height() for s in self._label_surf)
        y_off = self.rect.centery - total_h // 2
        for line in self._label_surf:
            x = self.rect.centerx - line.get_width() // 2
            surface.blit(line, (x, y_off))
            y_off += line.get_height()

    def handle_event(self, event):
        """Process pygame event. Returns True if button was activated."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.active = True
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_active = self.active
            self.active = False
            if was_active and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True

        elif event.type == pygame.FINGERDOWN:
            # For SDL2 with direct touch input
            w, h = pygame.display.get_surface().get_size()
            px = int(event.x * w)
            py = int(event.y * h)
            if self.rect.collidepoint(px, py):
                self.active = True
                return False

        elif event.type == pygame.FINGERUP:
            was_active = self.active
            self.active = False
            if was_active:
                w, h = pygame.display.get_surface().get_size()
                px = int(event.x * w)
                py = int(event.y * h)
                if self.rect.collidepoint(px, py):
                    if self.callback:
                        self.callback()
                    return True

        return False


class FpsCounter:
    """Simple FPS counter overlay."""

    def __init__(self, x=10, y=10):
        self.x = x
        self.y = y
        self._font = None

    def draw(self, surface, fps, extra_text=""):
        if self._font is None:
            self._font = pygame.font.Font(None, 24)
        text = f"FPS: {fps:.1f}"
        if extra_text:
            text += f" | {extra_text}"
        surf = self._font.render(text, True, COLORS["fps_text"])
        surface.blit(surf, (self.x, self.y))


class OverlayRenderer:
    """Draws detection results and geometry overlays on the camera frame."""

    @staticmethod
    def draw_detections(surface, detections, detector):
        """Draw bounding boxes and labels for all detections."""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            conf = det.confidence
            class_name = detector.get_class_name(det.class_id)

            # Bounding box
            pygame.draw.rect(surface, COLORS["ball_bbox"],
                             (x1, y1, x2 - x1, y2 - y1), 2)

            # Center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            pygame.draw.circle(surface, COLORS["ball_center"],
                               (cx, cy), 4)

            # Ground contact point (bottom center)
            gx = cx
            gy = y2
            pygame.draw.circle(surface, COLORS["accent_red"],
                               (gx, gy), 3)
            pygame.draw.line(surface, COLORS["accent_red"],
                             (gx - 6, gy), (gx + 6, gy), 2)

            # Label
            font = pygame.font.Font(None, 20)
            label = f"{class_name} {conf:.2f}"
            label_surf = font.render(label, True, COLORS["text_primary"])
            surface.blit(label_surf, (x1, y1 - 18))

    @staticmethod
    def draw_goal_polygon(surface, polygon_np):
        """Draw the scoring zone polygon overlay."""
        if polygon_np is None or len(polygon_np) < 3:
            return

        pts = polygon_np.reshape((-1, 2)).astype(int).tolist()

        # Filled semi-transparent
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(overlay, COLORS["goal_polygon"], pts)
        surface.blit(overlay, (0, 0))

        # Outline
        pygame.draw.polygon(surface, COLORS["goal_outline"], pts, 3)

        # Corner markers
        for i, pt in enumerate(pts):
            x, y = pt
            pygame.draw.circle(surface, COLORS["accent_yellow"], (x, y), 6)
            font = pygame.font.Font(None, 18)
            corner_label = font.render(str(i + 1), True, COLORS["text_primary"])
            surface.blit(corner_label, (x + 8, y - 8))

    @staticmethod
    def draw_goal_flash(surface):
        """Full-screen red flash for goal event."""
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((255, 0, 0, 80))
        surface.blit(overlay, (0, 0))

        font = pygame.font.Font(None, 80)
        text = font.render("GOAL!", True, COLORS["accent_red"])
        text_rect = text.get_rect(center=(surface.get_width() // 2,
                                          surface.get_height() // 2))
        surface.blit(text, text_rect)


def scale_coords(x1, y1, x2, y2, from_size, to_size):
    """Scale bounding box coordinates from one resolution to another."""
    fx, fy = from_size
    tx, ty = to_size
    sx, sy = tx / fx, ty / fy
    return (
        int(x1 * sx), int(y1 * sy),
        int(x2 * sx), int(y2 * sy),
    )
