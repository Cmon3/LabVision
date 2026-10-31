"""Morphotype clustering from measured features (Python decides, Muse describes)."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

# Fixed global scales computed on dev split (mean, std per feature).
# If not yet computed, fallback to defaults that approximate dev distribution.
# Features order: L_adj, a_adj, b_adj, circularity, log_diam, texture
# Computed via bench/compute_scales.py on dev split GT boxes with Otsu masks.
# Will be overwritten by config when available.
try:
    from labvision.config import MORPHOTYPE_SCALES as _SCALES
    from labvision.config import MORPHOTYPE_MEANS as _MEANS
    GLOBAL_MEANS = np.array(_MEANS, dtype=float)
    GLOBAL_SCALES = np.array(_SCALES, dtype=float)
except Exception:
    # fallback: rough estimates from synthetic + early dev run
    GLOBAL_MEANS = np.array([0.0, 0.0, 0.0, 0.85, 3.5, 2.0], dtype=float)
    GLOBAL_SCALES = np.array([12.0, 8.0, 8.0, 0.15, 0.4, 3.0], dtype=float)


def standardize(features: np.ndarray, means: np.ndarray | None = None, scales: np.ndarray | None = None) -> np.ndarray:
    m = GLOBAL_MEANS if means is None else np.asarray(means, dtype=float)
    s = GLOBAL_SCALES if scales is None else np.asarray(scales, dtype=float)
    s = np.where(s < 1e-6, 1.0, s)
    return (features - m) / s


def _merge_small_clusters(features_std: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Merge clusters with < max(3, 10% of N) into nearest cluster by centroid distance."""
    n = len(features_std)
    if n == 0:
        return labels
    thresh = max(3, int(np.ceil(0.1 * n)))
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) <= 1:
        return labels
    small = set(uniq[counts < thresh])
    if not small:
        return labels
    large = [u for u in uniq if u not in small]
    # compute centroids of large clusters
    centroids = {}
    for u in large:
        centroids[u] = features_std[labels == u].mean(axis=0)
    # for each point in small clusters, reassign to nearest large centroid
    new_labels = labels.copy()
    for idx in range(n):
        if labels[idx] in small:
            dists = {u: np.linalg.norm(features_std[idx] - centroids[u]) for u in large}
            best = min(dists, key=dists.get)  # type: ignore
            new_labels[idx] = best
    return new_labels


def cluster_plate(features: np.ndarray, threshold: float = 2.5, color_weight: float = 1.0, means=None, scales=None) -> np.ndarray:
    """Agglomerative clustering (average linkage) with threshold T.

    Returns int labels 0..K-1. Default is one cluster.
    """
    n = features.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    # standardize with fixed globals
    std = standardize(features, means, scales)

    # apply color weight to L,a,b dims (0,1,2)
    if color_weight != 1.0:
        std[:, 0:3] *= color_weight

    # if all points identical, return one cluster
    if np.allclose(std, std[0]):
        return np.zeros(n, dtype=int)

    # hierarchical clustering
    try:
        dists = pdist(std, metric="euclidean")
        # check for NaN/inf
        if not np.isfinite(dists).all():
            return np.zeros(n, dtype=int)
        Z = linkage(dists, method="average")
        labels = fcluster(Z, t=threshold, criterion="distance")
        # fcluster gives 1-indexed labels; convert to 0-indexed
        labels = labels - 1
        # merge small clusters
        labels = _merge_small_clusters(std, labels)
        # reindex to 0..K-1 contiguous
        uniq = np.unique(labels)
        remap = {old: new for new, old in enumerate(sorted(uniq))}
        labels = np.array([remap[v] for v in labels], dtype=int)
        return labels
    except Exception:
        return np.zeros(n, dtype=int)


def purity_flag(n_types: int) -> str:
    if n_types <= 1:
        return "Single morphotype (consistent with pure culture)"
    return f"{n_types} morphotypes detected (possible mixed culture or contamination): confirm with standard microbiological methods."


def describe_clusters(features: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Per-cluster summary: label char A,B.., count, mean diam, feature mean."""
    uniq = sorted(set(labels))
    out = []
    for i, u in enumerate(uniq):
        mask = labels == u
        count = int(mask.sum())
        # mean log_diam -> convert back to diam for display (exp)
        mean_log_diam = float(features[mask, 4].mean()) if mask.any() else 0.0
        mean_diam = float(np.exp(mean_log_diam))
        lab = chr(ord("A") + i)
        out.append(
            {
                "label": lab,
                "cluster_id": int(u),
                "count": count,
                "mean_log_diam": mean_log_diam,
                "mean_diameter_px": mean_diam,
                "feature_mean": features[mask].mean(axis=0).tolist() if mask.any() else [],
            }
        )
    return out
