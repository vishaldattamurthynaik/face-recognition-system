"""
Visualization utilities for drawing bounding boxes, facial landmarks, identity tags, and status HUDs.
"""
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.detector import FaceDetection
from src.matcher import MatchResult


# Theme Color Palette (BGR format)
COLOR_KNOWN = (70, 200, 70)      # Vibrant Emerald Green
COLOR_UNKNOWN = (60, 60, 235)     # Vibrant Crimson Red
COLOR_LANDMARK = (255, 200, 0)   # Cyan / Light Blue
COLOR_ENROLL = (235, 150, 50)    # Orange / Amber
COLOR_TEXT = (255, 255, 255)     # Pure White
COLOR_BG = (25, 25, 25)          # Dark Slate


def draw_detection_and_recognition(
    image: np.ndarray,
    detection: FaceDetection,
    match_result: Optional[MatchResult] = None,
    draw_landmarks: bool = True,
    show_similarity: bool = True
) -> np.ndarray:
    """
    Renders bounding box, 5-point facial landmarks, and identity pill tag on an image.

    Args:
        image: BGR numpy image.
        detection: FaceDetection containing bbox and landmarks.
        match_result: Optional MatchResult for identity and confidence.
        draw_landmarks: Whether to render the 5 facial landmarks.
        show_similarity: Whether to show cosine similarity score.

    Returns:
        Annotated BGR numpy image copy.
    """
    canvas = image.copy()
    x, y, w, h = detection.bbox

    # Choose color scheme based on match
    if match_result is None:
        color = COLOR_ENROLL
        label_text = f"Face: {detection.score:.2f}"
    elif match_result.is_known:
        color = COLOR_KNOWN
        if show_similarity:
            label_text = f"{match_result.name} ({match_result.similarity_score:.2f} | {match_result.confidence_pct:.0f}%)"
        else:
            label_text = match_result.name
    else:
        color = COLOR_UNKNOWN
        if show_similarity and match_result.similarity_score > 0:
            label_text = f"Unknown (Sim: {match_result.similarity_score:.2f})"
        else:
            label_text = "Unknown"

    # Draw stylish Corner Brackets & Box
    thickness = max(2, int(round(min(image.shape[:2]) / 350)))
    line_len = max(12, int(w * 0.2))

    # Base rectangle outline (translucent/thin)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, max(1, thickness - 1))

    # Bold corner accents
    # Top-Left
    cv2.line(canvas, (x, y), (x + line_len, y), color, thickness + 1)
    cv2.line(canvas, (x, y), (x, y + line_len), color, thickness + 1)
    # Top-Right
    cv2.line(canvas, (x + w, y), (x + w - line_len, y), color, thickness + 1)
    cv2.line(canvas, (x + w, y), (x + w, y + line_len), color, thickness + 1)
    # Bottom-Left
    cv2.line(canvas, (x, y + h), (x + line_len, y + h), color, thickness + 1)
    cv2.line(canvas, (x, y + h), (x, y + h - line_len), color, thickness + 1)
    # Bottom-Right
    cv2.line(canvas, (x + w, y + h), (x + w - line_len, y + h), color, thickness + 1)
    cv2.line(canvas, (x + w, y + h), (x + w, y + h - line_len), color, thickness + 1)

    # Draw 5 Facial Landmarks
    if draw_landmarks and detection.landmarks is not None:
        for lm in detection.landmarks:
            lx, ly = int(lm[0]), int(lm[1])
            cv2.circle(canvas, (lx, ly), max(2, thickness), COLOR_LANDMARK, -1)

    # Render Text Pill Tag Above Bounding Box
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.45, min(w, h) / 260.0)
    font_thick = 1 if font_scale < 0.7 else 2

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

    tag_y1 = max(0, y - text_h - 12)
    tag_y2 = y
    tag_x1 = x
    tag_x2 = x + text_w + 14

    # Background Pill
    cv2.rectangle(canvas, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    # Text
    cv2.putText(
        canvas,
        label_text,
        (tag_x1 + 7, tag_y2 - 6),
        font,
        font_scale,
        COLOR_TEXT,
        font_thick,
        cv2.LINE_AA
    )

    return canvas


def draw_hud(
    image: np.ndarray,
    fps: Optional[float] = None,
    num_faces: int = 0,
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Renders top info bar overlay.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    
    hud_h = 32
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    info_text = f"Faces: {num_faces}"
    if fps is not None:
        info_text += f" | FPS: {fps:.1f}"
    if threshold is not None:
        info_text += f" | Cosine Threshold: {threshold:.2f}"

    cv2.putText(
        canvas,
        info_text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )
    return canvas
