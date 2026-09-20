"""
Unified Command-Line Interface (CLI) for the Face Recognition Identification System.
Supports enrollment, identification, 1:1 verification, database management, benchmark evaluation, and web launcher.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import DEFAULT_COSINE_THRESHOLD, UNKNOWN_LABEL
from src.pipeline import FaceRecognitionPipeline
from evaluation.benchmark import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="FaceMatrix AI: Face Recognition & Identification System CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enroll", action="store_true", help="Enroll a new identity with a face image.")
    group.add_argument("--identify", action="store_true", help="Identify faces in a query image.")
    group.add_argument("--verify", action="store_true", help="1:1 verification between two face images.")
    group.add_argument("--evaluate", action="store_true", help="Run automated benchmark evaluation suite.")
    group.add_argument("--list", action="store_true", help="List all enrolled identities.")
    group.add_argument("--web", action="store_true", help="Launch interactive Streamlit Web UI.")

    # Options
    parser.add_argument("--image", type=str, help="Path to input image for enrollment or identification.")
    parser.add_argument("--image1", type=str, help="First image for 1:1 verification.")
    parser.add_argument("--image2", type=str, help="Second image for 1:1 verification.")
    parser.add_argument("--name", type=str, help="Full Name of the person to enroll.")
    parser.add_argument("--id", type=str, help="Optional unique ID for enrollment.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_COSINE_THRESHOLD, help="Cosine similarity threshold.")
    parser.add_argument("--output", type=str, help="Path to save annotated output image.")

    args = parser.parse_args()

    if args.web:
        print("[*] Launching Streamlit Web Application...")
        app_path = Path(__file__).resolve().parent / "app.py"
        subprocess.run(["streamlit", "run", str(app_path)])
        return

    if args.evaluate:
        run_full_benchmark(custom_threshold=args.threshold)
        return

    pipeline = FaceRecognitionPipeline(cosine_threshold=args.threshold)

    if args.list:
        people = pipeline.database.list_people()
        print("\n" + "=" * 50)
        print(f"  ENROLLED IDENTITIES DATABASE ({len(people)} Total)")
        print("=" * 50)
        if not people:
            print("  (Database is empty)")
        for p in people:
            print(f"  • {p.name} (ID: {p.person_id}) - {p.num_samples} sample(s)")
        print("=" * 50 + "\n")
        return

    if args.enroll:
        if not args.image or not args.name:
            print("[ERROR] Please provide both --image <path> and --name <person_name> for enrollment.")
            sys.exit(1)
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERROR] Could not read image file: {args.image}")
            sys.exit(1)
        ok, msg, person = pipeline.enroll_face(img, name=args.name, person_id=args.id)
        if ok:
            print(f"[SUCCESS] {msg}")
        else:
            print(f"[FAILED] {msg}")
        return

    if args.identify:
        if not args.image:
            print("[ERROR] Please provide --image <path> for identification.")
            sys.exit(1)
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERROR] Could not read image file: {args.image}")
            sys.exit(1)
        
        output = pipeline.recognize_faces(img, return_annotated=True)
        print(f"\n[Detection Result]: Found {output.num_faces} face(s)")
        for idx, match in enumerate(output.matches, 1):
            status = "KNOWN" if match.is_known else "UNKNOWN (REJECTED)"
            print(f"  Face #{idx}: [{status}] Name: '{match.name}' | Cosine Similarity: {match.similarity_score:.4f} | Confidence: {match.confidence_pct}%")
        
        if args.output:
            cv2.imwrite(args.output, output.annotated_image)
            print(f"[OK] Saved annotated image to: {args.output}")
        return

    if args.verify:
        if not args.image1 or not args.image2:
            print("[ERROR] Please provide both --image1 and --image2 for verification.")
            sys.exit(1)
        img1 = cv2.imread(args.image1)
        img2 = cv2.imread(args.image2)
        if img1 is None or img2 is None:
            print("[ERROR] Failed to read one or both verification images.")
            sys.exit(1)
        is_same, sim, dist = pipeline.verify_pair(img1, img2)
        verdict = "MATCH (Same Person)" if is_same else "NO MATCH (Different / Unknown)"
        print(f"\n[1:1 Verification Result]: {verdict}")
        print(f"  - Cosine Similarity: {sim:.4f} (Threshold: {pipeline.matcher.cosine_threshold:.2f})")
        print(f"  - Euclidean Distance: {dist:.4f}")
        return


if __name__ == "__main__":
    main()
