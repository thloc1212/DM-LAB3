from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import adjusted_rand_score

from utils import quadruplet_revenue as _quadruplet_revenue_impl
from utils import triplet_revenue as _triplet_revenue_impl


def compute_aari(Z_pred, Z_true, levels):
    score = 0
    for l in levels:
        pred = cut_tree(Z_pred, n_clusters=l).flatten()
        true = cut_tree(Z_true, n_clusters=l).flatten()
        score += adjusted_rand_score(true, pred)
    return score / len(levels)


def triplet_revenue(tree, triplets):
    """
    Cai T_rev(H, T) = sum ( |H(i v k)| - |H(i v j)| ).
    """
    return _triplet_revenue_impl(tree, triplets)


def quartet_revenue(tree, quadruplets):
    """
    Q_rev(H, Q) tuong tu voi (i, j, k, l).
    """
    return _quadruplet_revenue_impl(tree, quadruplets)
