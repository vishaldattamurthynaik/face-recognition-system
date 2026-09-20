"""
Face Embedder module using OpenCV's SFace (SphereFace/MobileFaceNet) deep metric learning model.
Extracts 128-dimensional L2-normalized feature vectors from aligned face crops.
"""
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SFACE_MODEL_PATH, EMBEDDING_DIM
from src.detector import FaceDetection
from models.download_models import ensure_models_exist


class FaceEmbedder:
    """
    SFace Deep Feature Extractor.
    Extracts high-dimensional identity embeddings from detected face regions.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = str(model_path or SFACE_MODEL_PATH)
        self.embedding_dim = EMBEDDING_DIM

        # Ensure model is downloaded
        ensure_models_exist()

        # Initialize SFace recognizer via OpenCV DNN
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def align_face(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Aligns and crops the face to canonical 112x112 resolution using 5-point facial landmarks.

        Args:
            image: Input image (BGR).
            detection: FaceDetection object containing landmarks and raw detection array.

        Returns:
            112x112 aligned face chip (BGR numpy array).
        """
        return self.recognizer.alignCrop(image, detection.raw)

    def extract_embedding(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Aligns face and extracts 128-d L2-normalized identity embedding.

        Args:
            image: Input image (BGR).
            detection: FaceDetection object.

        Returns:
            1D numpy array of shape (128,) with unit L2 norm.
        """
        aligned_face = self.align_face(image, detection)
        raw_feature = self.recognizer.feature(aligned_face)  # shape (1, 128)
        
        # Flatten and normalize to unit hypersphere: ||v||_2 = 1.0
        vec = raw_feature.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def extract_from_aligned(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts embedding directly from pre-aligned 112x112 face image.
        """
        raw_feature = self.recognizer.feature(aligned_face)
        vec = raw_feature.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
