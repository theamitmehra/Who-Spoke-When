"""
Agglomerative Hierarchical Clustering (AHC) for speaker identity assignment.
Uses cosine similarity on ECAPA-TDNN embeddings to cluster segments by speaker.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from loguru import logger


class SpeakerClusterer:
    """
    Agglomerative Hierarchical Clustering for speaker diarization.
    Supports automatic speaker count estimation via silhouette analysis.
    """

    def __init__(
        self,
        linkage_method: str = "average",
        distance_threshold: float = 0.55,
        min_speakers: int = 1,
        max_speakers: int = 10,
    ):
        self.linkage_method = linkage_method
        self.distance_threshold = distance_threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def _cosine_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        similarity = embeddings @ embeddings.T
        distance = np.clip(1.0 - similarity, 0.0, 2.0)
        return distance

    def _estimate_num_speakers(self, embeddings: np.ndarray, linkage_matrix: np.ndarray) -> int:
        n = len(embeddings)
        if n <= 2:
            return n

        min_k = max(2, self.min_speakers)
        upper_k = min(self.max_speakers, n - 1)

        best_k = min_k
        best_score = -1.0

        for k in range(min_k, upper_k + 1):
            labels = fcluster(linkage_matrix, k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue
            try:
                score = silhouette_score(embeddings, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        threshold_labels = fcluster(
            linkage_matrix,
            t=self.distance_threshold,
            criterion="distance",
        )
        k_threshold = len(np.unique(threshold_labels))
        k_threshold = int(np.clip(k_threshold, self.min_speakers, min(self.max_speakers, n)))

        # Be conservative to avoid severe over-segmentation in open-domain audio.
        if best_score < 0.08:
            chosen_k = k_threshold
        else:
            chosen_k = min(best_k, k_threshold) if k_threshold >= 2 else best_k

        chosen_k = int(np.clip(chosen_k, self.min_speakers, min(self.max_speakers, n)))

        logger.info(
            f"Optimal speaker count: {chosen_k} "
            f"(silhouette_k={best_k}, silhouette={best_score:.4f}, threshold_k={k_threshold})"
        )
        return chosen_k

    def cluster(
        self,
        embeddings: np.ndarray,
        num_speakers: Optional[int] = None,
    ) -> np.ndarray:
        """Cluster embeddings into speaker identities."""
        n = len(embeddings)

        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        dist_matrix = self._cosine_distance_matrix(embeddings)
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method=self.linkage_method)

        if num_speakers is not None:
            k = max(1, min(num_speakers, n))
        else:
            k = self._estimate_num_speakers(embeddings, Z)

        labels = fcluster(Z, k, criterion="maxclust") - 1
        return labels.astype(int)

    def merge_consecutive_same_speaker(
        self,
        segments: List[Tuple[float, float]],
        labels: np.ndarray,
        gap_tolerance: float = 0.3,
    ) -> List[Tuple[float, float, int]]:
        """Merge consecutive segments assigned to the same speaker."""
        if not segments:
            return []

        merged = []
        current_start, current_end = segments[0]
        current_label = labels[0]

        for i in range(1, len(segments)):
            seg_start, seg_end = segments[i]
            seg_label = labels[i]
            gap = seg_start - current_end

            if seg_label == current_label and gap <= gap_tolerance:
                current_end = seg_end
            else:
                merged.append((current_start, current_end, int(current_label)))
                current_start, current_end = seg_start, seg_end
                current_label = seg_label

        merged.append((current_start, current_end, int(current_label)))
        return merged
