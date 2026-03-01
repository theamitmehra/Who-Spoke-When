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
        distance_threshold: float = 0.7,
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

        best_k = self.min_speakers
        best_score = -1.0
        upper_k = min(self.max_speakers, n - 1)

        for k in range(max(2, self.min_speakers), upper_k + 1):
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

        logger.info(f"Optimal speaker count: {best_k} (silhouette={best_score:.4f})")
        return best_k

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
