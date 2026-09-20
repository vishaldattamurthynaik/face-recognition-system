"""
Face Matcher module for 1:N similarity search and 'Unknown' rejection.
Supports Cosine Similarity, L2 Distance, multi-candidate ranking, and margin analysis.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_L2_THRESHOLD,
    DEFAULT_METRIC,
    UNKNOWN_LABEL
)
from src.database import EnrolledFaceDatabase


@dataclass
class CandidateMatch:
    """
    Representation of a single candidate comparison.
    """
    person_id: str
    name: str
    cosine_similarity: float
    l2_distance: float


@dataclass
class MatchResult:
    """
    Final identification decision result.
    """
    person_id: Optional[str]
    name: str
    is_known: bool
    similarity_score: float     # Cosine similarity to top candidate [-1, 1]
    distance: float             # Euclidean distance to top candidate [0, 2]
    confidence_pct: float       # Calibrated confidence percentage [0, 100]
    margin: float               # Difference between Top-1 and Top-2 similarity
    all_candidates: List[CandidateMatch]


class FaceMatcher:
    """
    Performs 1:N biometric verification and identification against an enrolled database.
    """
    def __init__(
        self,
        database: EnrolledFaceDatabase,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
        l2_threshold: float = DEFAULT_L2_THRESHOLD,
        metric: str = DEFAULT_METRIC,
        use_centroids: bool = True
    ):
        self.database = database
        self.cosine_threshold = cosine_threshold
        self.l2_threshold = l2_threshold
        self.metric = metric.lower()
        self.use_centroids = use_centroids

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Matches a query embedding against all enrolled identities in the database.

        Args:
            query_embedding: 128-d unit-normalized numpy array.

        Returns:
            MatchResult containing identity decision, confidence, and candidate ranking.
        """
        # Ensure query is 1D float32 normalized
        q = query_embedding.flatten().astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        if self.use_centroids:
            matrix, pids, names = self.database.get_all_centroids_matrix()
        else:
            matrix, pids, names = self.database.get_all_embeddings_matrix()

        # Handle empty database
        if len(pids) == 0:
            return MatchResult(
                person_id=None,
                name=UNKNOWN_LABEL,
                is_known=False,
                similarity_score=0.0,
                distance=2.0,
                confidence_pct=0.0,
                margin=0.0,
                all_candidates=[]
            )

        # Vectorized Cosine Similarity: S = matrix @ q  (since both are L2 normalized)
        cosine_sims = np.dot(matrix, q)

        # Vectorized Euclidean Distance: D = sqrt(2 - 2 * cos_sim)
        # Numerical stability clip to [-1.0, 1.0]
        cosine_sims_clipped = np.clip(cosine_sims, -1.0, 1.0)
        l2_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine_sims_clipped))

        # Build candidate list
        candidates: List[CandidateMatch] = []
        for i in range(len(pids)):
            candidates.append(
                CandidateMatch(
                    person_id=pids[i],
                    name=names[i],
                    cosine_similarity=float(cosine_sims[i]),
                    l2_distance=float(l2_dists[i])
                )
            )

        # Sort candidates by similarity descending
        candidates.sort(key=lambda c: c.cosine_similarity, reverse=True)

        top_match = candidates[0]
        top_sim = top_match.cosine_similarity
        top_dist = top_match.l2_distance

        # Calculate Margin (Gap between Rank 1 and Rank 2)
        margin = (top_sim - candidates[1].cosine_similarity) if len(candidates) > 1 else top_sim

        # Rejection Decision Logic
        if self.metric == "cosine":
            is_known = top_sim >= self.cosine_threshold
        elif self.metric == "l2":
            is_known = top_dist <= self.l2_threshold
        else:
            is_known = top_sim >= self.cosine_threshold

        # Calibrated Confidence Score
        # For Cosine: Scale [0, 1] relative to threshold
        # If sim >= threshold: confidence in [70%, 100%]
        # If sim < threshold: confidence drops rapidly
        if is_known:
            # Linear scaling from threshold -> 1.0 mapping to 70% -> 99.9%
            score_range = max(1.0 - self.cosine_threshold, 1e-5)
            conf_pct = 70.0 + 29.9 * min(1.0, max(0.0, (top_sim - self.cosine_threshold) / score_range))
        else:
            # Below threshold: confidence of being the matched person is low
            conf_pct = max(0.0, (top_sim / max(self.cosine_threshold, 1e-5)) * 60.0)

        decision_name = top_match.name if is_known else UNKNOWN_LABEL
        decision_pid = top_match.person_id if is_known else None

        return MatchResult(
            person_id=decision_pid,
            name=decision_name,
            is_known=is_known,
            similarity_score=top_sim,
            distance=top_dist,
            confidence_pct=round(conf_pct, 1),
            margin=round(float(margin), 4),
            all_candidates=candidates
        )
