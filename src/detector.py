"""
Face Detector module utilizing OpenCV's YuNet ONNX model.
Extracts bounding boxes, confidence scores, and 5-point facial landmarks.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    YUNET_MODEL_PATH,
    DETECTION_CONF_THRESHOLD,
    DETECTION_NMS_THRESHOLD,
    DETECTION_TOP_K
)
from models.download_models import ensure_models_exist


@dataclass
class FaceDetection:
    """
    Data class representing a detected face.
    """
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray             # 5x2 array: [right_eye, left_eye, nose_tip, right_mouth, left_mouth]
    score: float                      # Face confidence score in [0, 1]
    raw: np.ndarray                   # Full 15-element detection array from YuNet


class FaceDetector:
    """
    YuNet Face Detector wrapper with dynamic resolution handling and landmark extraction.
    """
    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = DETECTION_CONF_THRESHOLD,
        nms_threshold: float = DETECTION_NMS_THRESHOLD,
        top_k: int = DETECTION_TOP_K
    ):
        self.model_path = str(model_path or YUNET_MODEL_PATH)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._current_size = (320, 320)

        # Ensure model is downloaded
        ensure_models_exist()

        # Initialize YuNet detector via OpenCV DNN
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self._current_size,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detects faces in a BGR or RGB image.

        Args:
            image: Input image as numpy array (H, W, C).

        Returns:
            List of FaceDetection objects sorted by confidence score (descending).
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        if (w, h) != self._current_size:
            self._current_size = (w, h)
            self.detector.setInputSize((w, h))

        # Perform detection
        _, faces = self.detector.detect(image)

        results: List[FaceDetection] = []
        if faces is None or len(faces) == 0:
            return results

        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, fw, fh = map(int, face[0:4])
            landmarks = face[4:14].reshape((5, 2))
            score = float(face[14])

            # Clip bounding box to image boundaries
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            fw = max(1, min(fw, w - x))
            fh = max(1, min(fh, h - y))

            results.append(
                FaceDetection(
                    bbox=(x, y, fw, fh),
                    landmarks=landmarks,
                    score=score,
                    raw=face
                )
            )

        # Sort by confidence score descending
        results.sort(key=lambda f: f.score, reverse=True)
        return results

    def extract_face_crop(self, image: np.ndarray, detection: FaceDetection, margin: float = 0.1) -> np.ndarray:
        """
        Extracts a cropped face image with optional padding margin.
        """
        h, w = image.shape[:2]
        x, y, fw, fh = detection.bbox

        dx = int(fw * margin)
        dy = int(fh * margin)

        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(w, x + fw + dx)
        y2 = min(h, y + fh + dy)

        return image[y1:y2, x1:x2].copy()
