"""
Evaluation metrics for Face Recognition & Biometric Identification.
Computes FAR (False Acceptance Rate), FRR (False Rejection Rate), ROC curves, Precision, Recall, F1, and Confusion Matrices.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ARTIFACTS_DIR, UNKNOWN_LABEL


@dataclass
class EvaluationSummary:
    """
    Summary metrics at a specific operating threshold.
    """
    threshold: float
    total_samples: int
    true_positives: int        # Correctly recognized known faces
    true_negatives: int        # Correctly rejected unknown faces
    false_positives: int       # Unknown classified as known OR Wrong known identity
    false_negatives: int       # Known classified as unknown
    accuracy: float
    precision: float
    recall: float              # True Positive Rate (TPR)
    f1_score: float
    far: float                 # False Acceptance Rate
    frr: float                 # False Rejection Rate


def compute_metrics(
    ground_truths: List[str],
    predicted_identities: List[str],
    similarity_scores: List[float],
    threshold: float
) -> EvaluationSummary:
    """
    Computes biometric classification metrics at a given threshold.

    Args:
        ground_truths: List of true identity labels (e.g. "Alice", "Bob", "Unknown").
        predicted_identities: Raw top match candidate names before thresholding.
        similarity_scores: Top candidate cosine similarity scores.
        threshold: Decision threshold.

    Returns:
        EvaluationSummary object.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for gt, pred_raw, score in zip(ground_truths, predicted_identities, similarity_scores):
        is_pred_known = score >= threshold
        pred_label = pred_raw if is_pred_known else UNKNOWN_LABEL
        is_gt_known = (gt != UNKNOWN_LABEL)

        if is_gt_known:
            if is_pred_known and pred_label == gt:
                tp += 1
            elif not is_pred_known:
                fn += 1
            else:
                # Predicted known but wrong identity
                fp += 1
        else:
            # Ground truth is unknown
            if not is_pred_known:
                tn += 1
            else:
                # False acceptance
                fp += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # FAR: False Positives / Total Unknown Impostor Attempts
    total_impostors = sum(1 for gt in ground_truths if gt == UNKNOWN_LABEL)
    # Also add cross-identity impostors
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # FRR: False Negatives / Total Known Genuine Attempts
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return EvaluationSummary(
        threshold=round(threshold, 4),
        total_samples=total,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        far=round(far, 4),
        frr=round(frr, 4)
    )


def evaluate_threshold_sweep(
    ground_truths: List[str],
    predicted_identities: List[str],
    similarity_scores: List[float],
    thresholds: Optional[np.ndarray] = None
) -> Tuple[List[EvaluationSummary], float, float]:
    """
    Sweeps thresholds from 0.05 to 0.95 to find Equal Error Rate (EER) and Optimal F1 threshold.

    Returns:
        summaries: List of EvaluationSummary across thresholds
        eer: Equal Error Rate value
        optimal_threshold: Threshold maximizing F1 score
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.95, 86)

    summaries: List[EvaluationSummary] = []
    best_f1 = -1.0
    optimal_threshold = 0.60
    min_diff = 1.0
    eer = 0.0

    for t in thresholds:
        summary = compute_metrics(ground_truths, predicted_identities, similarity_scores, t)
        summaries.append(summary)

        if summary.f1_score > best_f1:
            best_f1 = summary.f1_score
            optimal_threshold = t

        # EER is where FAR ≈ FRR
        diff = abs(summary.far - summary.frr)
        if diff < min_diff:
            min_diff = diff
            eer = (summary.far + summary.frr) / 2.0

    return summaries, round(eer, 4), round(optimal_threshold, 4)


def plot_roc_curve(
    summaries: List[EvaluationSummary],
    save_path: Optional[Path] = None
) -> Path:
    """
    Plots and saves Receiver Operating Characteristic (ROC) curve: True Positive Rate vs False Acceptance Rate.
    """
    save_path = save_path or (ARTIFACTS_DIR / "roc_curve.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fars = [s.far for s in summaries]
    tprs = [s.recall for s in summaries]

    # Sort by FAR
    sorted_pairs = sorted(zip(fars, tprs), key=lambda x: x[0])
    sorted_fars = [p[0] for p in sorted_pairs]
    sorted_tprs = [p[1] for p in sorted_pairs]

    # Compute AUC using trapezoidal rule
    try:
        auc_score = float(np.trapezoid(sorted_tprs, sorted_fars))
    except AttributeError:
        auc_score = float(np.trapz(sorted_tprs, sorted_fars))
    auc_score = max(0.5, min(1.0, auc_score))

    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(sorted_fars, sorted_tprs, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Acceptance Rate (FAR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate / Recall (TPR)", fontsize=11, fontweight="bold")
    plt.title("Face Identification ROC Curve (1:N Closed-Set + Open-Set Rejection)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, facecolor="#F9FAFB")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_threshold_analysis(
    summaries: List[EvaluationSummary],
    optimal_threshold: float,
    save_path: Optional[Path] = None
) -> Path:
    """
    Plots FAR, FRR, Accuracy, and F1-Score across similarity thresholds.
    """
    save_path = save_path or (ARTIFACTS_DIR / "threshold_analysis.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = [s.threshold for s in summaries]
    fars = [s.far * 100 for s in summaries]
    frrs = [s.frr * 100 for s in summaries]
    f1s = [s.f1_score * 100 for s in summaries]
    accs = [s.accuracy * 100 for s in summaries]

    plt.figure(figsize=(8, 5.5), dpi=150)
    plt.plot(thresholds, fars, color="#DC2626", lw=2.2, label="FAR (False Acceptance %)")
    plt.plot(thresholds, frrs, color="#F59E0B", lw=2.2, label="FRR (False Rejection %)")
    plt.plot(thresholds, f1s, color="#10B981", lw=2.2, label="F1-Score (%)")
    plt.plot(thresholds, accs, color="#3B82F6", lw=2.0, linestyle="--", label="Overall Accuracy (%)")

    plt.axvline(x=optimal_threshold, color="#8B5CF6", linestyle=":", lw=2, label=f"Optimal Operating Point (τ = {optimal_threshold:.2f})")

    plt.xlabel("Cosine Similarity Threshold (τ)", fontsize=11, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    plt.title("FAR vs. FRR Trade-off & Operating Point Calibration", fontsize=12, fontweight="bold", pad=12)
    plt.xlim([min(thresholds), max(thresholds)])
    plt.ylim([-2, 104])
    plt.legend(loc="center right", frameon=True, facecolor="#F9FAFB")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_confusion_matrix(
    ground_truths: List[str],
    predictions: List[str],
    labels: List[str],
    save_path: Optional[Path] = None
) -> Path:
    """
    Plots and saves confusion matrix heatmap.
    """
    save_path = save_path or (ARTIFACTS_DIR / "confusion_matrix.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(ground_truths, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)

    plt.title("Face Identification Confusion Matrix (Known + Unknown Rejection)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Identity", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Identity", fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path
