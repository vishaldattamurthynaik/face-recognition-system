"""
Configuration settings for the Face Recognition Identification System.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ENROLLED_DIR = DATA_DIR / "enrolled"
TEST_KNOWN_DIR = DATA_DIR / "test_known"
TEST_UNKNOWN_DIR = DATA_DIR / "test_unknown"
DATABASE_PATH = DATA_DIR / "enrolled_database.json"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, ENROLLED_DIR, TEST_KNOWN_DIR, TEST_UNKNOWN_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model File Paths
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Model Download URLs (Official OpenCV Zoo with mirrors)
MODEL_URLS = {
    "yunet": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ],
    "sface": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ]
}

# Detection Hyper-parameters
DETECTION_CONF_THRESHOLD = 0.60  # Minimum confidence score for face detection
DETECTION_NMS_THRESHOLD = 0.30   # Non-maximum suppression threshold
DETECTION_TOP_K = 5000           # Top K detections before NMS

# Face Recognition & Matching Hyper-parameters
# Default Cosine Similarity Threshold (Cosine Similarity range: -1 to 1, higher is closer)
# SFace default recommended threshold: ~0.363 for standard pairs, ~0.60 for high-precision identification
DEFAULT_COSINE_THRESHOLD = 0.60

# Default Euclidean (L2) Distance Threshold (L2 distance range: 0 to 2, lower is closer)
# SFace default recommended threshold: ~1.128
DEFAULT_L2_THRESHOLD = 1.128

# Matching Metric: "cosine" or "l2"
DEFAULT_METRIC = "cosine"

# Unknown Label
UNKNOWN_LABEL = "Unknown"

# Embedding Dimension for SFace
EMBEDDING_DIM = 128
