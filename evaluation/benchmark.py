"""
Benchmark Evaluation Suite for Face Recognition Identification System.
Evaluates recognition accuracy, FAR, FRR, ROC-AUC, and generates artifact visual plots.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_DIR,
    ENROLLED_DIR,
    TEST_KNOWN_DIR,
    TEST_UNKNOWN_DIR,
    DEFAULT_COSINE_THRESHOLD,
    UNKNOWN_LABEL,
    ARTIFACTS_DIR
)
from src.pipeline import FaceRecognitionPipeline
from evaluation.generate_sample_dataset import prepare_benchmark_dataset, AUTHENTIC_DATASET
from evaluation.metrics import (
    compute_metrics,
    evaluate_threshold_sweep,
    plot_roc_curve,
    plot_threshold_analysis,
    plot_confusion_matrix
)


def run_full_benchmark(custom_threshold: float = DEFAULT_COSINE_THRESHOLD) -> Dict:
    """
    Executes end-to-end benchmark evaluation.
    """
    print("=" * 70)
    print("      FACE RECOGNITION SYSTEM: BENCHMARK & EVALUATION SUITE       ")
    print("=" * 70)

    # Step 1: Ensure models and dataset are ready
    prepare_benchmark_dataset()

    # Step 2: Initialize pipeline with isolated test database
    test_db_path = DATA_DIR / "benchmark_db.json"
    if test_db_path.exists():
        test_db_path.unlink()

    pipeline = FaceRecognitionPipeline(
        cosine_threshold=custom_threshold,
        db_path=test_db_path
    )
    pipeline.database.clear()

    # Step 3: Enroll known subjects
    print("\n[*] Enrolling known identities into benchmark database...")
    known_identities = AUTHENTIC_DATASET["known"]

    for key, data in known_identities.items():
        display_name = data["name"]
        img_files = [f for f in ENROLLED_DIR.glob(f"{key}*.jpg") if f.is_file()]
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                ok, msg, _ = pipeline.enroll_face(img, name=display_name, person_id=key)
                if ok:
                    print(f"  + {msg}")

    print(f"[OK] Total Enrolled Identities: {pipeline.database.count()}")

    # Step 4: Run test queries
    print("\n[*] Evaluating on Known and Unknown test queries...")
    ground_truths: List[str] = []
    top_candidates: List[str] = []
    top_similarities: List[float] = []
    test_file_paths: List[str] = []

    # 4a: Test Known Queries
    for key, data in known_identities.items():
        display_name = data["name"]
        known_files = [f for f in TEST_KNOWN_DIR.glob(f"{key}*.jpg") if f.is_file()]
        for kf in known_files:
            img = cv2.imread(str(kf))
            if img is None:
                continue
            out = pipeline.recognize_faces(img, return_annotated=False)
            if len(out.detections) > 0 and len(out.matches) > 0:
                match = out.matches[0]
                ground_truths.append(display_name)
                top_cand_name = match.all_candidates[0].name if match.all_candidates else UNKNOWN_LABEL
                top_candidates.append(top_cand_name)
                top_similarities.append(match.similarity_score)
                test_file_paths.append(str(kf.name))

    # 4b: Test Unknown Impostor Queries
    unknown_files = [f for f in TEST_UNKNOWN_DIR.glob("*.jpg") if f.is_file()]
    for uf in unknown_files:
        img = cv2.imread(str(uf))
        if img is None:
            continue
        out = pipeline.recognize_faces(img, return_annotated=False)
        if len(out.detections) > 0 and len(out.matches) > 0:
            match = out.matches[0]
            ground_truths.append(UNKNOWN_LABEL)
            top_cand_name = match.all_candidates[0].name if match.all_candidates else UNKNOWN_LABEL
            top_candidates.append(top_cand_name)
            top_similarities.append(match.similarity_score)
            test_file_paths.append(str(uf.name))

    total_queries = len(ground_truths)
    num_known_queries = sum(1 for g in ground_truths if g != UNKNOWN_LABEL)
    num_unknown_queries = sum(1 for g in ground_truths if g == UNKNOWN_LABEL)

    print(f"[OK] Total Test Queries: {total_queries} (Known: {num_known_queries}, Unknown: {num_unknown_queries})")

    # Step 5: Threshold Sweep & Calibration Analysis
    summaries, eer, optimal_thresh = evaluate_threshold_sweep(
        ground_truths=ground_truths,
        predicted_identities=top_candidates,
        similarity_scores=top_similarities
    )

    # Step 6: Generate Visual Plots
    roc_path = plot_roc_curve(summaries, ARTIFACTS_DIR / "roc_curve.png")
    thresh_plot_path = plot_threshold_analysis(summaries, optimal_thresh, ARTIFACTS_DIR / "threshold_analysis.png")

    # Predictions at Optimal Threshold
    opt_summary = compute_metrics(ground_truths, top_candidates, top_similarities, threshold=optimal_thresh)
    opt_predictions = [
        cand if score >= optimal_thresh else UNKNOWN_LABEL
        for cand, score in zip(top_candidates, top_similarities)
    ]
    all_classes = [data["name"] for data in known_identities.values()] + [UNKNOWN_LABEL]
    cm_path = plot_confusion_matrix(ground_truths, opt_predictions, labels=all_classes, save_path=ARTIFACTS_DIR / "confusion_matrix.png")

    # Step 7: Print Metrics Table
    print("\n" + "=" * 70)
    print("                    PERFORMANCE EVALUATION SUMMARY                    ")
    print("=" * 70)
    print(f"Optimal Operating Threshold:  tau* = {optimal_thresh:.2f}")
    print(f"Equal Error Rate (EER):       {eer * 100:.2f}%")
    print(f"Overall Accuracy:             {opt_summary.accuracy * 100:.2f}%")
    print(f"Precision:                    {opt_summary.precision * 100:.2f}%")
    print(f"Recall (TPR):                 {opt_summary.recall * 100:.2f}%")
    print(f"F1-Score:                     {opt_summary.f1_score * 100:.2f}%")
    print(f"False Acceptance Rate (FAR):  {opt_summary.far * 100:.2f}%")
    print(f"False Rejection Rate (FRR):   {opt_summary.frr * 100:.2f}%")
    print("-" * 70)
    print(f"True Positives (Known Correct):       {opt_summary.true_positives}")
    print(f"True Negatives (Unknown Rejected):    {opt_summary.true_negatives}")
    print(f"False Positives (Impostors accepted): {opt_summary.false_positives}")
    print(f"False Negatives (Known missed):        {opt_summary.false_negatives}")
    print("=" * 70)

    print("\n[Artifacts Generated]:")
    print(f"  - ROC Curve:           {roc_path}")
    print(f"  - Threshold Analysis:  {thresh_plot_path}")
    print(f"  - Confusion Matrix:    {cm_path}")

    # Generate Markdown Table for README
    md_table = """
| Threshold (tau) | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | FAR (%) | FRR (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for t_val in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        s = compute_metrics(ground_truths, top_candidates, top_similarities, threshold=t_val)
        marker = " *(Optimal)*" if abs(t_val - optimal_thresh) < 0.05 else ""
        md_table += f"| **{t_val:.2f}**{marker} | {s.accuracy*100:.1f}% | {s.precision*100:.1f}% | {s.recall*100:.1f}% | {s.f1_score*100:.1f}% | {s.far*100:.1f}% | {s.frr*100:.1f}% |\n"

    print("\n[Markdown Results Table]:\n" + md_table)

    return {
        "optimal_threshold": optimal_thresh,
        "eer": eer,
        "accuracy": opt_summary.accuracy,
        "precision": opt_summary.precision,
        "recall": opt_summary.recall,
        "f1_score": opt_summary.f1_score,
        "far": opt_summary.far,
        "frr": opt_summary.frr,
        "md_table": md_table
    }


if __name__ == "__main__":
    run_full_benchmark()
