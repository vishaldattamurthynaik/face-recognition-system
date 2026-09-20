"""
Dataset Generator & Fetcher for Face Recognition Benchmark.
Prepares authentic face dataset containing known (enrolled) identities and unknown (non-enrolled) identities.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, ENROLLED_DIR, TEST_KNOWN_DIR, TEST_UNKNOWN_DIR


AUTHENTIC_DATASET = {
    # Known Enrolled Subjects
    "known": {
        "barack_obama": {
            "name": "Barack Obama",
            "urls": [
                "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
                "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg"
            ]
        },
        "joe_biden": {
            "name": "Joe Biden",
            "urls": [
                "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg"
            ]
        },
        "alex_lacamoire": {
            "name": "Alex Lacamoire",
            "urls": [
                "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/alex-lacamoire.png"
            ]
        },
        "lin_manuel_miranda": {
            "name": "Lin-Manuel Miranda",
            "urls": [
                "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/lin-manuel-miranda.png"
            ]
        }
    },
    # Unknown Impostor Subjects (Never Enrolled in database)
    "unknown": {
        "unknown_subject_1": {
            "name": "Unknown Subject 1",
            "urls": [
                "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/two_people.jpg"
            ]
        }
    }
}


def download_image(url: str, output_path: Path) -> bool:
    """
    Downloads image with User-Agent header.
    """
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=12) as response:
            data = response.read()
            if len(data) > 1000:
                with open(output_path, "wb") as f:
                    f.write(data)
                return True
    except Exception as e:
        print(f"[!] Warning: Failed to fetch {url}: {e}")
    return False


def apply_augmentations(img: np.ndarray, aug_type: str) -> np.ndarray:
    """
    Applies realistic perturbations: brightness change, rotation, blur, contrast.
    """
    aug = img.copy()
    if aug_type == "bright":
        aug = cv2.convertScaleAbs(aug, alpha=1.2, beta=20)
    elif aug_type == "dark":
        aug = cv2.convertScaleAbs(aug, alpha=0.8, beta=-25)
    elif aug_type == "blur":
        aug = cv2.GaussianBlur(aug, (5, 5), 1.2)
    elif aug_type == "rot_left":
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif aug_type == "rot_right":
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -4, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return aug


def prepare_benchmark_dataset():
    """
    Builds the authentic benchmark dataset.
    """
    print("[*] Preparing Authentic Face Recognition Benchmark Dataset...")
    for d in [ENROLLED_DIR, TEST_KNOWN_DIR, TEST_UNKNOWN_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Known Subjects
    for key, data in AUTHENTIC_DATASET["known"].items():
        urls = data["urls"]
        base_path = ENROLLED_DIR / f"{key}_base.jpg"
        download_image(urls[0], base_path)

        base_img = cv2.imread(str(base_path))
        if base_img is not None:
            # Enrolled sample 2 (mild variation)
            cv2.imwrite(str(ENROLLED_DIR / f"{key}_sample2.jpg"), apply_augmentations(base_img, "bright"))

            # Test Known Probes
            if len(urls) > 1:
                probe1_path = TEST_KNOWN_DIR / f"{key}_probe_different_photo.jpg"
                download_image(urls[1], probe1_path)
            
            cv2.imwrite(str(TEST_KNOWN_DIR / f"{key}_probe_dark.jpg"), apply_augmentations(base_img, "dark"))
            cv2.imwrite(str(TEST_KNOWN_DIR / f"{key}_probe_blur.jpg"), apply_augmentations(base_img, "blur"))
            cv2.imwrite(str(TEST_KNOWN_DIR / f"{key}_probe_rot.jpg"), apply_augmentations(base_img, "rot_right"))

    # 2. Unknown Impostor Subjects
    for key, data in AUTHENTIC_DATASET["unknown"].items():
        urls = data["urls"]
        unk_path = TEST_UNKNOWN_DIR / f"{key}_probe1.jpg"
        download_image(urls[0], unk_path)

        unk_img = cv2.imread(str(unk_path))
        if unk_img is not None:
            cv2.imwrite(str(TEST_UNKNOWN_DIR / f"{key}_probe2_bright.jpg"), apply_augmentations(unk_img, "bright"))
            cv2.imwrite(str(TEST_UNKNOWN_DIR / f"{key}_probe3_dark.jpg"), apply_augmentations(unk_img, "dark"))

    print(f"[SUCCESS] Dataset prepared successfully in {DATA_DIR}")


if __name__ == "__main__":
    prepare_benchmark_dataset()
