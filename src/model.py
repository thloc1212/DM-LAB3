"""Model-level clustering helpers used by the notebooks.

The implementations intentionally mirror the lightweight fallbacks in
``src/utils.py`` so the notebooks can import a single stable surface.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS


def compute_adds3(num_objects: int, triplets):
    """Build a similarity matrix from triplets.

    Each triplet ``(i, j, k)`` means object ``i`` is closer to ``j`` than to
    ``k``. The returned matrix is symmetric, non-negative, and has a boosted
    diagonal so it can be used directly with average linkage.
    """
    S = np.zeros((num_objects, num_objects), dtype=float)
    triplets = np.asarray(triplets, dtype=np.int64)
    if triplets.size == 0:
        np.fill_diagonal(S, 1.0)
        return S

    for i, j, k in triplets:
        S[i, j] += 1.0
        S[j, i] += 1.0
        S[i, k] -= 1.0
        S[k, i] -= 1.0

    S = (S + S.T) / 2.0
    S -= S.min()
    np.fill_diagonal(S, S.max() + 1.0)
    return S


def compute_adds4(num_objects: int, quadruplets):
    """Build a similarity matrix from quadruplets.

    Each quadruplet ``(i, j, k, l)`` means pair ``(i, j)`` is closer than
    pair ``(k, l)``. The implementation is a simple pair-support heuristic:
    reward the preferred pair and penalize the rejected pair.
    """
    S = np.zeros((num_objects, num_objects), dtype=float)
    quadruplets = np.asarray(quadruplets, dtype=np.int64)
    if quadruplets.size == 0:
        np.fill_diagonal(S, 1.0)
        return S

    for i, j, k, l in quadruplets:
        S[i, j] += 1.0
        S[j, i] += 1.0
        S[k, l] -= 1.0
        S[l, k] -= 1.0

    S = (S + S.T) / 2.0
    S -= S.min()
    np.fill_diagonal(S, S.max() + 1.0)
    return S


def average_linkage_from_similarity(S):
    """Convert a similarity matrix into an average-linkage hierarchy."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square similarity matrix")
    S = (S + S.T) / 2.0
    D = np.max(S) - S
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method="average")


def adds3_al(num_objects, triplets):
    S = compute_adds3(num_objects, triplets)
    tree = average_linkage_from_similarity(S)
    return tree, S


def adds4_al(num_objects, quadruplets):
    S = compute_adds4(num_objects, quadruplets)
    tree = average_linkage_from_similarity(S)
    return tree, S


def tste_al(num_objects, triplets, n_components=10, **tste_kwargs):
    """Proxy tSTE + average linkage using metric MDS and cosine similarity."""
    triplets = np.asarray(triplets, dtype=np.int64)
    C = np.zeros((num_objects, num_objects), dtype=float)
    for i, j, k in triplets:
        C[i, j] += 1.0
        C[j, i] += 1.0
        C[i, k] -= 0.5
        C[k, i] -= 0.5

    C -= C.min()
    np.fill_diagonal(C, C.max())
    D = np.max(C) - C
    np.fill_diagonal(D, 0.0)

    emb = MDS(
        n_components=min(int(n_components), max(1, num_objects - 1)),
        dissimilarity="precomputed",
        random_state=tste_kwargs.get("random_state", 42),
    ).fit_transform(D)
    S = cosine_similarity(emb)
    tree = average_linkage_from_similarity(S)
    return tree, S


def mulk3_al(num_objects, triplets, **kwargs):
    """MulK3-style similarity estimation followed by average linkage."""
    triplets = np.asarray(triplets, dtype=np.int64)
    W = np.zeros((num_objects, num_objects), dtype=float)
    C = np.zeros((num_objects, num_objects), dtype=float)
    for i, j, k in triplets:
        W[i, j] += 1.0
        W[j, i] += 1.0
        C[i, j] += 1.0
        C[i, k] += 1.0
        C[j, i] += 1.0
        C[k, i] += 1.0

    S = np.divide(W, C, out=np.zeros_like(W), where=C > 0)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    tree = average_linkage_from_similarity(S)
    return tree, S


def fourk_al(num_objects, quadruplets, **kwargs):
    """4K-AL baseline based on the same quadruplet similarity heuristic."""
    return run_4k_al(num_objects, quadruplets)


def run_4k_al(num_objects, quadruplets):
    """Baseline for quadruplets using the same pair-support heuristic as AddS4."""
    S = compute_adds4(num_objects, quadruplets)
    tree = average_linkage_from_similarity(S)
    return tree, S