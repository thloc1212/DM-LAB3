"""Evaluation metrics for comparison-based clustering.

This module provides functions to calculate the Averaged Adjusted Rand Index (AARI)
and Revenue scores (Triplet and Quadruplet) based on the hierarchical tree structure.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score

# Import the core implementations (assumed to be optimized C/Cython or Numba functions)
from utils import quadruplet_revenue as _quadruplet_revenue_impl
from utils import triplet_revenue as _triplet_revenue_impl


def compute_aari(Z_pred: np.ndarray, y_true: np.ndarray, max_clusters: int = 8) -> float:
    """Compute the Averaged Adjusted Rand Index (AARI).

    AARI measures how well the predicted hierarchical tree matches the ground-truth 
    labels across multiple levels of the hierarchy. It cuts the dendrogram into 
    `l` clusters (for l from 2 to max_clusters), computes the ARI against the 
    ground-truth labels, and averages the scores.

    Args:
        Z_pred (np.ndarray): The predicted linkage matrix (dendrogram) of shape (n-1, 4).
        y_true (np.ndarray): 1D array of ground-truth cluster labels.
        max_clusters (int, optional): The maximum number of clusters to evaluate. 
                                      Defaults to 8 (as used in the Planted Model).

    Returns:
        float: The averaged ARI score.
    """
    y_true = np.asarray(y_true).flatten()
    score = 0.0
    
    # Calculate ARI for tree cuts from 2 up to max_clusters
    levels = range(2, max_clusters + 1)
    
    for l in levels:
        # Cut the predicted tree to get 'l' flat clusters
        pred_labels = cut_tree(Z_pred, n_clusters=l).flatten()
        score += adjusted_rand_score(y_true, pred_labels)
        
    return score / len(levels)


def triplet_revenue(tree: np.ndarray, triplets: np.ndarray) -> float:
    """Calculate the Triplet Revenue for a given hierarchical tree.

    For a triplet (i, j, k) meaning "i is closer to j than to k", the tree H 
    satisfies this if the Lowest Common Ancestor (LCA) of i and j is lower 
    (smaller subtree size) than the LCA of i and k.
    
    Formula: T_rev(H, T) = sum( |H(i ∨ k)| - |H(i ∨ j)| )
    where |H(x ∨ y)| is the size (number of leaves) of the subtree rooted 
    at the LCA of x and y.

    Args:
        tree (np.ndarray): The linkage matrix representing the hierarchy.
        triplets (np.ndarray): Array of triplets of shape (n_triplets, 3).

    Returns:
        float: The total triplet revenue score.
    """
    return _triplet_revenue_impl(tree, triplets)


def quartet_revenue(tree: np.ndarray, quadruplets: np.ndarray) -> float:
    """Calculate the Quadruplet Revenue for a given hierarchical tree.

    For a quadruplet (i, j, k, l) meaning "pair (i, j) is more similar than 
    pair (k, l)", the tree H satisfies this if the LCA of i and j is lower 
    than the LCA of k and l.
    
    Formula: Q_rev(H, Q) = sum( |H(k ∨ l)| - |H(i ∨ j)| )

    Args:
        tree (np.ndarray): The linkage matrix representing the hierarchy.
        quadruplets (np.ndarray): Array of quadruplets of shape (n_quadruplets, 4).

    Returns:
        float: The total quadruplet revenue score.
    """
    return _quadruplet_revenue_impl(tree, quadruplets)