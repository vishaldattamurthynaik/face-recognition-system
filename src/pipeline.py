"""
End-to-end Face Recognition Pipeline orchestrator.
Integrates detection, alignment, embedding extraction, database management, similarity matching, and visualization.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_L2_THRESHOLD,
    DEFAULT_METRIC,
    ENROLLED_DIR
)
from src.detector import FaceDetector, FaceDetection
from src.embedder import FaceEmbedder
from src.database import EnrolledFaceDatabase, EnrolledPerson
from src.matcher import FaceMatcher, MatchResult
from src.visualizer import draw_detection_and_recognition, draw_hud


@dataclass
class RecognitionOutput:
    """
    Complete output package for an image processed through the pipeline.
    """
    annotated_image: np.ndarray
    detections: List[FaceDetection]
    matches: List[MatchResult]
    num_faces: int
    num_known: int
    num_unknown: int


class FaceRecognitionPipeline:
    """
    High-level API for face enrollment, multi-face identification, and 1:1 face verification.
    """
    def __init__(
        self,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        l2_threshold: float = DEFAULT_L2_THRESHOLD,
        metric: str = DEFAULT_METRIC,
        conf_threshold: float = 0.60,
        db_path: Optional[Path] = None
    ):
        self.detector = FaceDetector(conf_threshold=conf_threshold)
        self.embedder = FaceEmbedder()
        self.database = EnrolledFaceDatabase(db_path=db_path)
        self.matcher = FaceMatcher(
            database=self.database,
            cosine_threshold=cosine_threshold,
            l2_threshold=l2_threshold,
            metric=metric
        )

    def set_threshold(self, cosine_threshold: float, l2_threshold: Optional[float] = None):
        """
        Dynamically updates the matching threshold.
        """
        self.matcher.cosine_threshold = float(cosine_threshold)
        if l2_threshold is not None:
            self.matcher.l2_threshold = float(l2_threshold)

    def enroll_face(
        self,
        image: np.ndarray,
        name: str,
        person_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        save_crop_dir: Optional[Path] = None
    ) -> Tuple[bool, str, Optional[EnrolledPerson]]:
        """
        Detects face, extracts embedding, and enrolls person into database.

        Args:
            image: BGR numpy array.
            name: Identity name.
            person_id: Optional unique string ID.
            metadata: Optional dictionary of attributes.
            save_crop_dir: Optional folder to save enrolled face crop.

        Returns:
            Tuple of (success: bool, message: str, enrolled_record: Optional[EnrolledPerson])
        """
        if image is None or image.size == 0:
            return False, "Invalid or empty image provided.", None

        detections = self.detector.detect(image)
        if len(detections) == 0:
            return False, "No face detected in the enrollment image.", None

        if len(detections) > 1:
            # We select the highest-confidence / largest face, but warn
            print(f"[*] Warning: Multiple faces ({len(detections)}) found. Using highest-confidence face.")

        target_detection = detections[0]
        embedding = self.embedder.extract_embedding(image, target_detection)

        # Save to database
        person = self.database.enroll(
            name=name,
            embedding=embedding,
            person_id=person_id,
            metadata=metadata
        )

        # Optionally save aligned face crop to disk
        target_dir = save_crop_dir or (ENROLLED_DIR / person.person_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        aligned_chip = self.embedder.align_face(image, target_detection)
        crop_filename = target_dir / f"sample_{person.num_samples}.jpg"
        cv2.imwrite(str(crop_filename), aligned_chip)

        return True, f"Successfully enrolled {name} (Sample #{person.num_samples})", person

    def recognize_faces(
        self,
        image: np.ndarray,
        return_annotated: bool = True,
        draw_landmarks: bool = True,
        show_similarity: bool = True,
        fps: Optional[float] = None
    ) -> RecognitionOutput:
        """
        Runs full detection + embedding + matching pipeline on an image or video frame.

        Args:
            image: BGR numpy array.
            return_annotated: If True, renders bounding boxes and badges on the image.
            draw_landmarks: If True, renders facial landmarks.
            show_similarity: If True, renders cosine similarity values.
            fps: Optional FPS counter for HUD.

        Returns:
            RecognitionOutput object with detections, match results, and annotated frame.
        """
        if image is None or image.size == 0:
            return RecognitionOutput(
                annotated_image=np.zeros((100, 100, 3), dtype=np.uint8),
                detections=[],
                matches=[],
                num_faces=0,
                num_known=0,
                num_unknown=0
            )

        detections = self.detector.detect(image)
        matches: List[MatchResult] = []
        canvas = image.copy() if return_annotated else image

        num_known = 0
        num_unknown = 0

        for det in detections:
            embedding = self.embedder.extract_embedding(image, det)
            match_res = self.matcher.match(embedding)
            matches.append(match_res)

            if match_res.is_known:
                num_known += 1
            else:
                num_unknown += 1

            if return_annotated:
                canvas = draw_detection_and_recognition(
                    image=canvas,
                    detection=det,
                    match_result=match_res,
                    draw_landmarks=draw_landmarks,
                    show_similarity=show_similarity
                )

        if return_annotated:
            canvas = draw_hud(
                image=canvas,
                fps=fps,
                num_faces=len(detections),
                threshold=self.matcher.cosine_threshold
            )

        return RecognitionOutput(
            annotated_image=canvas,
            detections=detections,
            matches=matches,
            num_faces=len(detections),
            num_known=num_known,
            num_unknown=num_unknown
        )

    def verify_pair(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        1:1 Direct verification between two face images.

        Returns:
            Tuple of (is_same_person: bool, cosine_similarity: float, l2_distance: float)
        """
        dets1 = self.detector.detect(image1)
        dets2 = self.detector.detect(image2)

        if len(dets1) == 0 or len(dets2) == 0:
            return False, 0.0, 2.0

        emb1 = self.embedder.extract_embedding(image1, dets1[0])
        emb2 = self.embedder.extract_embedding(image2, dets2[0])

        sim = float(np.dot(emb1, emb2))
        dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * min(1.0, max(-1.0, sim)))))

        is_match = sim >= self.matcher.cosine_threshold
        return is_match, sim, dist
