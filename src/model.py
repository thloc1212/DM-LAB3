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


def compute_adds3(num_objects: int, triplets: np.ndarray) -> np.ndarray:
    """Build a similarity matrix from triplets using the AddS3 heuristic.

    Each triplet (i, j, k) implies that object `i` is more similar to `j` 
    than to `k`. The heuristic rewards the (i, j) pair by adding 1 and 
    penalizes the (i, k) pair by subtracting 1.

    Args:
        num_objects (int): Total number of unique objects in the dataset.
        triplets (np.ndarray): Array of shape (n_triplets, 3) where each row is (i, j, k).

    Returns:
        np.ndarray: A symmetric, non-negative similarity matrix of shape (num_objects, num_objects)
        with a boosted diagonal, ready for average linkage.
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


def compute_adds4(num_objects: int, quadruplets: np.ndarray) -> np.ndarray:
    """Build a similarity matrix from quadruplets using the AddS4 heuristic.

    Each quadruplet (i, j, k, l) implies that the pair (i, j) is more similar 
    than the pair (k, l). The heuristic adds 1 to the preferred pair and 
    subtracts 1 from the rejected pair.

    Args:
        num_objects (int): Total number of unique objects in the dataset.
        quadruplets (np.ndarray): Array of shape (n_quadruplets, 4) where each row is (i, j, k, l).

    Returns:
        np.ndarray: A symmetric, non-negative similarity matrix of shape (num_objects, num_objects).
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


def average_linkage_from_similarity(S: np.ndarray) -> np.ndarray:
    """Convert a similarity matrix into an Average-Linkage hierarchical dendrogram.

    Converts similarity scores to distance scores using D = max(S) - S, zeros 
    out the diagonal, and applies scipy's average linkage algorithm.

    Args:
        S (np.ndarray): A square, symmetric similarity matrix.

    Returns:
        np.ndarray: The hierarchical clustering encoded as a linkage matrix.

    Raises:
        ValueError: If S is not a 2D square matrix.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square similarity matrix")
    S = (S + S.T) / 2.0
    D = np.max(S) - S
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method="average")


def adds3_al(num_objects: int, triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Execute the full AddS3-AL clustering pipeline.

    Args:
        num_objects (int): Total number of objects.
        triplets (np.ndarray): Triplet comparisons.

    Returns:
        tuple: (linkage_tree, similarity_matrix)
    """
    S = compute_adds3(num_objects, triplets)
    tree = average_linkage_from_similarity(S)
    return tree, S


def adds4_al(num_objects: int, quadruplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Execute the full AddS4-AL clustering pipeline.

    Args:
        num_objects (int): Total number of objects.
        quadruplets (np.ndarray): Quadruplet comparisons.

    Returns:
        tuple: (linkage_tree, similarity_matrix)
    """
    S = compute_adds4(num_objects, quadruplets)
    tree = average_linkage_from_similarity(S)
    return tree, S


def tste_al(num_objects: int, triplets: np.ndarray, n_components: int = 10, **tste_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Proxy implementation of tSTE + Average Linkage.

    NOTE: True tSTE requires gradient descent optimization over a Student-t distribution.
    This function implements a lightweight proxy by computing a similarity heuristic,
    converting it to distances, embedding via Multidimensional Scaling (MDS), 
    and then computing Cosine similarity in the embedded space.

    Args:
        num_objects (int): Total number of objects.
        triplets (np.ndarray): Triplet comparisons.
        n_components (int, optional): Dimensions for MDS embedding. Defaults to 10.

    Returns:
        tuple: (linkage_tree, similarity_matrix)
    """
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


def mulk3_al(num_objects: int, triplets: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Execute the MulK3-AL clustering pipeline.

    Estimates similarity by tracking the ratio of "wins" to "total comparisons"
    for each object pair based on the provided triplets.

    Args:
        num_objects (int): Total number of objects.
        triplets (np.ndarray): Triplet comparisons.

    Returns:
        tuple: (linkage_tree, similarity_matrix)
    """
    triplets = np.asarray(triplets, dtype=np.int64)
    W = np.zeros((num_objects, num_objects), dtype=float) # Wins matrix
    C = np.zeros((num_objects, num_objects), dtype=float) # Comparisons matrix
    
    for i, j, k in triplets:
        # (i, j) is the winning pair
        W[i, j] += 1.0
        W[j, i] += 1.0
        C[i, j] += 1.0
        C[j, i] += 1.0
        
        # (i, k) is the losing pair, so it only gets a comparison count, no win
        C[i, k] += 1.0
        C[k, i] += 1.0

    # S = Wins / Comparisons
    S = np.divide(W, C, out=np.zeros_like(W), where=C > 0)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    
    tree = average_linkage_from_similarity(S)
    return tree, S


def fourk_al(num_objects: int, quadruplets: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Execute the 4K-AL clustering pipeline.

    Calculates similarity by tracking the ratio of "wins" to "total comparisons"
    for quadruplets. If pair (i, j) is closer than (k, l), (i, j) gets a win, 
    but both pairs get a comparison count.

    Args:
        num_objects (int): Total number of objects.
        quadruplets (np.ndarray): Array of shape (n_quadruplets, 4).

    Returns:
        tuple: (linkage_tree, similarity_matrix)
    """
    quadruplets = np.asarray(quadruplets, dtype=np.int64)
    W = np.zeros((num_objects, num_objects), dtype=float) # Wins matrix
    C = np.zeros((num_objects, num_objects), dtype=float) # Comparisons matrix
    
    for i, j, k, l in quadruplets:
        # (i, j) wins
        W[i, j] += 1.0
        W[j, i] += 1.0
        C[i, j] += 1.0
        C[j, i] += 1.0
        
        # (k, l) loses
        C[k, l] += 1.0
        C[l, k] += 1.0

    # S = Wins / Comparisons
    S = np.divide(W, C, out=np.zeros_like(W), where=C > 0)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    
    tree = average_linkage_from_similarity(S)
    return tree, S