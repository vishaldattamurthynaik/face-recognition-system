"""
Unit Tests for Face Recognition Identification System.
Tests detector, embedder, database, matcher, and unknown rejection.
"""
import unittest
from pathlib import Path
import numpy as np
import cv2

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_COSINE_THRESHOLD, UNKNOWN_LABEL
from models.download_models import ensure_models_exist
from src.detector import FaceDetector, FaceDetection
from src.embedder import FaceEmbedder
from src.database import EnrolledFaceDatabase
from src.matcher import FaceMatcher
from src.pipeline import FaceRecognitionPipeline
from evaluation.generate_sample_dataset import create_synthetic_face


class TestFaceRecognitionSystem(unittest.TestCase):
    """
    Test suite for core biometric pipeline components.
    """
    @classmethod
    def setUpClass(cls):
        # Ensure ONNX models are present
        ensure_models_exist()
        cls.test_face_img1 = create_synthetic_face(seed=10)
        cls.test_face_img2 = create_synthetic_face(seed=20)

    def test_01_detector(self):
        detector = FaceDetector()
        dets = detector.detect(self.test_face_img1)
        self.assertIsInstance(dets, list)
        self.assertGreaterEqual(len(dets), 1, "YuNet should detect synthetic face")
        first_det = dets[0]
        self.assertEqual(len(first_det.bbox), 4)
        self.assertEqual(first_det.landmarks.shape, (5, 2))
        self.assertGreater(first_det.score, 0.5)

    def test_02_embedder(self):
        detector = FaceDetector()
        embedder = FaceEmbedder()
        dets = detector.detect(self.test_face_img1)
        self.assertGreaterEqual(len(dets), 1)
        
        emb = embedder.extract_embedding(self.test_face_img1, dets[0])
        self.assertEqual(emb.shape, (128,))
        # Check L2 Unit Norm
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=4, msg="Embedding vector must be unit L2 normalized")

    def test_03_database_crud_and_centroid(self):
        test_db_file = Path(__file__).resolve().parent / "test_scratch_db.json"
        if test_db_file.exists():
            test_db_file.unlink()

        db = EnrolledFaceDatabase(db_path=test_db_file)
        self.assertEqual(db.count(), 0)

        # Enroll sample 1
        vec1 = np.random.randn(128).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)
        p1 = db.enroll(name="Test User", embedding=vec1, person_id="user_1")
        self.assertEqual(db.count(), 1)
        self.assertEqual(p1.num_samples, 1)

        # Enroll sample 2 for same person -> updates centroid
        vec2 = np.random.randn(128).astype(np.float32)
        vec2 /= np.linalg.norm(vec2)
        p2 = db.enroll(name="Test User", embedding=vec2, person_id="user_1")
        self.assertEqual(db.count(), 1)
        self.assertEqual(p2.num_samples, 2)
        self.assertAlmostEqual(np.linalg.norm(p2.centroid), 1.0, places=4)

        # Reload from disk
        db_reloaded = EnrolledFaceDatabase(db_path=test_db_file)
        self.assertEqual(db_reloaded.count(), 1)
        person = db_reloaded.get_person("user_1")
        self.assertIsNotNone(person)
        self.assertEqual(person.name, "Test User")

        # Delete
        db_reloaded.delete_person("user_1")
        self.assertEqual(db_reloaded.count(), 0)

        if test_db_file.exists():
            test_db_file.unlink()

    def test_04_matcher_and_unknown_rejection(self):
        test_db_file = Path(__file__).resolve().parent / "test_scratch_db2.json"
        if test_db_file.exists():
            test_db_file.unlink()

        db = EnrolledFaceDatabase(db_path=test_db_file)
        
        # Enroll Known Vector
        known_vec = np.zeros(128, dtype=np.float32)
        known_vec[0] = 1.0  # [1, 0, 0, ...]
        db.enroll(name="Target Person", embedding=known_vec, person_id="target")

        matcher = FaceMatcher(database=db, cosine_threshold=0.60)

        # Test 1: Identical query -> must match
        query_known = known_vec.copy()
        res_known = matcher.match(query_known)
        self.assertTrue(res_known.is_known)
        self.assertEqual(res_known.name, "Target Person")
        self.assertAlmostEqual(res_known.similarity_score, 1.0, places=4)

        # Test 2: Orthogonal/Dissimilar query -> must be REJECTED as Unknown
        query_unknown = np.zeros(128, dtype=np.float32)
        query_unknown[1] = 1.0  # [0, 1, 0, ...] (similarity = 0.0)
        res_unknown = matcher.match(query_unknown)
        self.assertFalse(res_unknown.is_known, "Dissimilar vector must be rejected")
        self.assertEqual(res_unknown.name, UNKNOWN_LABEL)
        self.assertAlmostEqual(res_unknown.similarity_score, 0.0, places=4)

        if test_db_file.exists():
            test_db_file.unlink()

    def test_05_end_to_end_pipeline(self):
        test_db_file = Path(__file__).resolve().parent / "test_scratch_pipeline_db.json"
        if test_db_file.exists():
            test_db_file.unlink()

        pipeline = FaceRecognitionPipeline(db_path=test_db_file)
        
        # Enroll synthetic face
        ok, msg, person = pipeline.enroll_face(self.test_face_img1, name="Synthetic Subject")
        self.assertTrue(ok)
        self.assertIsNotNone(person)

        # Recognize on same image
        output = pipeline.recognize_faces(self.test_face_img1, return_annotated=True)
        self.assertEqual(output.num_faces, 1)
        self.assertGreaterEqual(output.num_known, 1)
        self.assertEqual(output.matches[0].name, "Synthetic Subject")

        if test_db_file.exists():
            test_db_file.unlink()


if __name__ == "__main__":
    unittest.main()
