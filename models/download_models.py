"""
Utility to download pre-trained ONNX models (YuNet and SFace) for the Face Recognition System.
$0 Spend, 100% open-source from OpenCV Zoo.
"""
import os
import sys
from pathlib import Path
import requests

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, MODEL_URLS


def download_file(urls, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Downloads a file from a list of mirror URLs to the destination path.
    """
    if destination.exists() and destination.stat().st_size > 10000:
        print(f"[OK] Model already exists: {destination.name} ({destination.stat().st_size / (1024*1024):.2f} MB)")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)

    for url in urls:
        print(f"[*] Downloading {destination.name} from {url} ...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                temp_dest = destination.with_suffix(".tmp")
                with open(temp_dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB)", end="")
                print()
                temp_dest.replace(destination)
                print(f"[SUCCESS] Downloaded {destination.name} successfully.")
                return True
            else:
                print(f"[!] HTTP error {response.status_code} for {url}")
        except Exception as e:
            print(f"[!] Failed from {url}: {e}")

    print(f"[ERROR] Could not download {destination.name} from any available source.")
    return False


def ensure_models_exist():
    """
    Ensures both YuNet and SFace ONNX models are present in the models directory.
    """
    yunet_ok = download_file(MODEL_URLS["yunet"], YUNET_MODEL_PATH)
    sface_ok = download_file(MODEL_URLS["sface"], SFACE_MODEL_PATH)
    if not (yunet_ok and sface_ok):
        raise RuntimeError("Failed to verify or download required ONNX models.")
    return True


if __name__ == "__main__":
    ensure_models_exist()
