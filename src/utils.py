"""Utility functions for data loading, processing, and evaluating comparison-based clustering."""

import zipfile
from pathlib import Path

import cblearn.datasets as cbd
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cut_tree, linkage, to_tree
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def safe_call(func, *args, **kwargs):
    """Safely call a function if it exists. Returns None if func is None."""
    if func is None:
        return None
    return func(*args, **kwargs)


def naive_similarity(n: int, triplets: np.ndarray) -> np.ndarray:
    """Build a naive similarity matrix by only counting co-occurrences.
    
    Args:
        n (int): Number of objects.
        triplets (np.ndarray): Array of triplets.
        
    Returns:
        np.ndarray: A normalized symmetric similarity matrix.
    """
    S = np.zeros((n, n), dtype=float)
    triplets = np.asarray(triplets, dtype=np.int64)
    for i, j, _ in triplets:
        S[i, j] += 1
        S[j, i] += 1
    if S.max() > 0:
        S = S / S.max()
    np.fill_diagonal(S, 1.0)
    return S


def generate_triplets_from_similarity(S: np.ndarray, num_samples: int, random_state: int = 0) -> np.ndarray:
    """Sample valid triplets from a ground-truth similarity matrix.
    
    For 3 randomly chosen items (i, j, k), if S[i, j] >= S[i, k], 
    the triplet is recorded as (i, j, k). Otherwise, (i, k, j).
    """
    rng = np.random.default_rng(random_state)
    n = S.shape[0]
    triplets = np.empty((num_samples, 3), dtype=np.int64)
    for t in range(num_samples):
        i, j, k = rng.choice(n, size=3, replace=False)
        triplets[t] = (i, j, k) if S[i, j] >= S[i, k] else (i, k, j)
    return triplets


def _require_file(path: str | Path) -> str:
    """Ensure a file exists and return its path as a string."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return str(p)


def load_zoo_features(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the UCI Zoo dataset.
    
    Returns:
        tuple: (X_features, y_labels)
    """
    path = _require_file(file_path)
    df = pd.read_csv(path)
    X = df.drop(columns=["animal_name", "class_type"]).to_numpy(dtype=float)
    y = df["class_type"].to_numpy()
    return X, y


def load_glass_features(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the UCI Glass dataset."""
    path = _require_file(file_path)
    df = pd.read_csv(path)
    X = df.drop(columns=["Type"]).to_numpy(dtype=float)
    y = df["Type"].to_numpy()
    return X, y


def load_mnist_features(train_path: str, test_path: str, n_per_class: int = 200, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Load a subset of the MNIST dataset and apply PCA.
    
    Args:
        train_path (str): Path to MNIST train CSV.
        test_path (str): Path to MNIST test CSV.
        n_per_class (int): Number of images to sample per digit (0-9).
        
    Returns:
        tuple: (X_pca_2d, y_labels)
    """
    train_file = _require_file(train_path)
    test_file = _require_file(test_path)
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df = pd.concat([df_train, df_test], ignore_index=True)

    y = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    
    rng = np.random.default_rng(random_state)
    idx = []
    for c in range(10):
        ids = np.where(y == c)[0]
        pick = rng.choice(ids, size=min(n_per_class, len(ids)), replace=False)
        idx.append(pick)
    idx = np.concatenate(idx)
    
    X_sub = X[idx]
    y_sub = y[idx]
    
    X_2d = PCA(n_components=2, random_state=random_state).fit_transform(X_sub)
    X_2d = X_2d / (np.abs(X_2d).max(axis=0, keepdims=True) + 1e-12)
    return X_2d, y_sub


def check_dataset(X: np.ndarray, y: np.ndarray, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate dataset shape and class distribution."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"{dataset_name}: X must be 2D")
    if len(X) != len(y):
        raise ValueError(f"{dataset_name}: X and y size mismatch")
    if len(np.unique(y)) < 2:
        raise ValueError(f"{dataset_name}: need at least 2 classes")
    return X, y


def build_cosine_triplets(X: np.ndarray, k_factor: float = 1.0, num_triplets: int = None, noise_rate: float = 0.05, random_state: int = 0) -> np.ndarray:
    """Generate triplets based on Cosine similarity with optional noise.
    
    Args:
        X (np.ndarray): Feature matrix.
        k_factor (float): Multiplier for n^2 comparisons.
        num_triplets (int): Explicit number of triplets (overrides k_factor).
        noise_rate (float): Percentage of triplets to flip (inject noise).
        
    Returns:
        np.ndarray: Array of triplets (n, 3).
    """
    n = X.shape[0]
    if num_triplets is not None:
        m = int(max(1, num_triplets))
    else:
        if k_factor is None:
            raise ValueError("k_factor must not be None when num_triplets is not provided")
        m = int(max(1, round(k_factor * (n ** 2))))
        
    S = cosine_similarity(X)
    triplets = generate_triplets_from_similarity(S, m, random_state=random_state)

    n_noisy = int(noise_rate * m)
    if n_noisy > 0:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(m, size=n_noisy, replace=False)
        # Flip the winner and loser to create noise
        triplets[idx, 1], triplets[idx, 2] = triplets[idx, 2].copy(), triplets[idx, 1].copy()
    return triplets


# =====================================================================
# SIMILARITY & CLUSTERING CORE (Mirrors model.py)
# =====================================================================

def compute_adds3_similarity(num_objects: int, triplets: np.ndarray) -> np.ndarray:
    """AddS3 Heuristic: +1 for winner pair, -1 for loser pair."""
    S = np.zeros((num_objects, num_objects), dtype=float)
    for i, j, k in triplets:
        S[i, j] += 1.0
        S[j, i] += 1.0
        S[i, k] -= 1.0
        S[k, i] -= 1.0
    S = (S + S.T) / 2.0
    S -= S.min()
    np.fill_diagonal(S, S.max() + 1.0)
    return S


def compute_mulk3_similarity(num_objects: int, triplets: np.ndarray) -> np.ndarray:
    """MulK3 Heuristic: Win / Comparisons ratio."""
    W = np.zeros((num_objects, num_objects), dtype=float)
    C = np.zeros((num_objects, num_objects), dtype=float)
    for i, j, k in triplets:
        W[i, j] += 1.0; W[j, i] += 1.0
        C[i, j] += 1.0; C[j, i] += 1.0
        C[i, k] += 1.0; C[k, i] += 1.0
    S = np.divide(W, C, out=np.zeros_like(W), where=C > 0)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    return S


def compute_adds4_similarity(num_objects: int, quadruplets: np.ndarray) -> np.ndarray:
    """AddS4 Heuristic: +1 for winning pair, -1 for losing pair (FIXED)."""
    S = np.zeros((num_objects, num_objects), dtype=np.float64)
    quadruplets = np.asarray(quadruplets, dtype=np.int64)
    for i, j, k, l in quadruplets:
        S[i, j] += 1.0; S[j, i] += 1.0
        S[k, l] -= 1.0; S[l, k] -= 1.0
    S = (S + S.T) / 2.0
    S -= S.min()
    np.fill_diagonal(S, S.max() + 1.0)
    return S


def average_linkage_from_similarity(S: np.ndarray) -> np.ndarray:
    """Convert similarity matrix to distance matrix and apply average linkage."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Similarity matrix must be square")
    S = (S + S.T) / 2.0
    D = np.max(S) - S
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method="average")


def tste_al_proxy(num_objects: int, triplets: np.ndarray, n_components: int = 10, random_state: int = 42) -> tuple:
    """Proxy implementation of tSTE using MDS and Cosine Similarity."""
    C = np.zeros((num_objects, num_objects), dtype=float)
    for i, j, k in triplets:
        C[i, j] += 1.0; C[j, i] += 1.0
        C[i, k] -= 0.5; C[k, i] -= 0.5
    C -= C.min()
    np.fill_diagonal(C, C.max())
    D = np.max(C) - C
    np.fill_diagonal(D, 0.0)
    
    emb = MDS(
        n_components=min(n_components, num_objects - 1),
        dissimilarity="precomputed",
        random_state=random_state,
    ).fit_transform(D)
    S = cosine_similarity(emb)
    Z = average_linkage_from_similarity(S)
    return Z, S


def run_adds4_al(num_objects: int, quadruplets: np.ndarray) -> tuple:
    """Run full AddS4 + Average Linkage pipeline."""
    S = compute_adds4_similarity(num_objects, quadruplets)
    Z = average_linkage_from_similarity(S)
    return Z, S


def run_4k_al(num_objects: int, quadruplets: np.ndarray) -> tuple:
    """Run full 4K + Average Linkage pipeline (FIXED logical bug).
    
    Properly calculates W/C ratio for 4-item comparisons instead of 
    trimming down to triplets.
    """
    quadruplets = np.asarray(quadruplets, dtype=np.int64)
    W = np.zeros((num_objects, num_objects), dtype=float)
    C = np.zeros((num_objects, num_objects), dtype=float)
    for i, j, k, l in quadruplets:
        W[i, j] += 1.0; W[j, i] += 1.0
        C[i, j] += 1.0; C[j, i] += 1.0
        C[k, l] += 1.0; C[l, k] += 1.0
    S = np.divide(W, C, out=np.zeros_like(W), where=C > 0)
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    Z = average_linkage_from_similarity(S)
    return Z, S


# =====================================================================
# EVALUATION METRICS (AARI, Revenue)
# =====================================================================

def _build_parent_and_size(linkage_matrix: np.ndarray) -> tuple:
    """Helper: Build tree parent pointers and subtree sizes for LCA queries."""
    root, nodes = to_tree(linkage_matrix, rd=True)
    parent = {}
    sizes = {}
    for node in nodes:
        sizes[node.id] = node.count
        if node.left is not None:
            parent[node.left.id] = node.id
        if node.right is not None:
            parent[node.right.id] = node.id
    return root.id, parent, sizes


def _lca_size(a: int, b: int, root_id: int, parent: dict, sizes: dict) -> int:
    """Helper: Find the Lowest Common Ancestor (LCA) size of two leaves."""
    anc = set()
    x = a
    anc.add(x)
    while x in parent:
        x = parent[x]
        anc.add(x)
    y = b
    while y not in anc and y in parent:
        y = parent[y]
    if y not in anc:
        y = root_id
    return sizes[y]


def triplet_revenue(linkage_matrix: np.ndarray, triplets: np.ndarray) -> float:
    """Calculate Triplet Revenue: sum( |H(i v k)| - |H(i v j)| )."""
    if linkage_matrix is None or len(triplets) == 0: return 0.0
    root_id, parent, sizes = _build_parent_and_size(linkage_matrix)
    rev = 0.0
    for i, j, k in triplets:
        rev += _lca_size(i, k, root_id, parent, sizes) - _lca_size(i, j, root_id, parent, sizes)
    return float(rev)


def quadruplet_revenue(linkage_matrix: np.ndarray, quads: np.ndarray) -> float:
    """Calculate Quadruplet Revenue: sum( |H(l v k)| - |H(i v j)| )."""
    if linkage_matrix is None or quads.size == 0: return 0.0
    root_id, parent, sizes = _build_parent_and_size(linkage_matrix)
    rev = 0.0
    for i, j, l, k in quads:
        rev += _lca_size(l, k, root_id, parent, sizes) - _lca_size(i, j, root_id, parent, sizes)
    return float(rev)


def compute_aari(linkage_matrix: np.ndarray, labels: np.ndarray, max_clusters: int = 10) -> float:
    """Calculate Averaged Adjusted Rand Index (AARI)."""
    y = np.asarray(labels).reshape(-1)
    if y.dtype == object:
        _, y = np.unique(y, return_inverse=True)
    n = len(y)
    if n < 3:
        return np.nan
    
    k_max = int(min(max_clusters, n - 1))
    ks = list(range(2, k_max + 1))
    if not ks:
        return np.nan
        
    scores = []
    for k in ks:
        pred = cut_tree(linkage_matrix, n_clusters=k).reshape(-1)
        scores.append(adjusted_rand_score(y, pred))
    return float(np.mean(scores))


# =====================================================================
# EXTERNAL DATASET LOADERS
# =====================================================================

def get_cblearn_array(fetcher_func):
    """Safely unpack data from cblearn functions."""
    data = fetcher_func()
    if hasattr(data, "data"): return data.data
    if hasattr(data, "triplets"): return data.triplets
    if hasattr(data, "queries"): return data.queries
    return data[list(data.keys())[0]]


def triplets_to_quadruplets(triplets: np.ndarray) -> np.ndarray:
    """Convert Triplets (i, j, k) to Quadruplets (i, j, i, k)."""
    T = np.asarray(triplets, dtype=np.int64)
    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError(f"Expected shape (n, 3), got {T.shape}")
    return np.column_stack([T[:, 0], T[:, 1], T[:, 0], T[:, 2]]).astype(np.int64)


def load_cblearn_triplets(dataset_name: str, num_triplets: int = 50000, random_state: int = 0):
    """General loader for cblearn datasets."""
    name = dataset_name.replace("-", "_")
    fn_map = {
        "car": "fetch_car_similarity", "food": "fetch_food_similarity",
        "imagenet": "fetch_imagenet_similarity", "material": "fetch_material_similarity",
        "musician": "fetch_musician_similarity", "nature": "fetch_nature_scene_similarity",
        "things": "fetch_things_similarity", "vogue": "fetch_vogue_cover_similarity",
    }
    if name not in fn_map:
        raise RuntimeError(f"Dataset {dataset_name} not supported")

    fn = getattr(cbd, fn_map[name])
    try: obj = fn(return_triplets=True)
    except TypeError: obj = fn()

    if isinstance(obj, np.ndarray): triplets = np.asarray(obj)
    elif isinstance(obj, tuple): triplets = np.asarray(obj[0])
    elif hasattr(obj, "triplet"): triplets = np.asarray(obj.triplet)
    elif hasattr(obj, "data"):
        data = np.asarray(obj.data)
        if data.ndim == 2 and data.shape[1] >= 3: triplets = data[:, :3]
        else: raise ValueError(f"Unsupported shape: {data.shape}")
    else: raise TypeError(f"Unsupported return type")

    triplets = np.asarray(triplets, dtype=np.int64)
    if num_triplets is not None and len(triplets) > num_triplets:
        rng = np.random.default_rng(random_state)
        triplets = triplets[rng.choice(len(triplets), size=num_triplets, replace=False)]
    object_ids = np.arange(int(triplets.max()) + 1 if triplets.size else 0)
    return triplets.astype(np.int64), object_ids.astype(np.int64)


def load_offline_triplets(dataset_name: str, zip_path: str):
    """Load triplets from an offline zip file."""
    zip_path = Path(zip_path)
    if not zip_path.exists(): raise FileNotFoundError(f"Missing file: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        filename = next(name for name in zf.namelist() if f"{dataset_name}_triplets.txt" in name)
        with zf.open(filename) as f:
            raw_data = [line.decode("utf-8").strip().split() for line in f if line.strip()]
    
    unique_items = np.unique(raw_data)
    item_to_id = {item: idx for idx, item in enumerate(unique_items)}
    standard_triplets = []
    for row in raw_data:
        i, j, k = item_to_id[row[0]], item_to_id[row[1]], item_to_id[row[2]]
        standard_triplets.append([j, k, i])
        standard_triplets.append([k, j, i])
    return np.asarray(standard_triplets, dtype=np.int64)


def load_car_triplets():
    raw_data = get_cblearn_array(lambda: cbd.fetch_car_similarity())
    standard_triplets = []
    for i, j, k in raw_data:
        standard_triplets.extend([[j, i, k], [k, i, j]])
    return np.asarray(standard_triplets, dtype=np.int64)


def load_imagenet_triplets():
    raw_rows = get_cblearn_array(lambda: cbd.fetch_imagenet_similarity())
    if raw_rows.shape[1] == 3: return np.asarray(raw_rows, dtype=np.int64)
    standard_triplets = []
    for row in raw_rows:
        i0, i1, i2 = row[0], row[1], row[2]
        for k in range(2, 8): standard_triplets.append([i0, i1, row[k]])
        for k in range(3, 8): standard_triplets.append([i0, i2, row[k]])
    return np.asarray(standard_triplets, dtype=np.int64)


def load_palmer_penguins_2d(random_state=42):
    """Load Penguins dataset and reduce to 2D via PCA for toy examples."""
    df = sns.load_dataset("penguins").dropna()
    X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].values.astype(np.float32)
    species_map = {s: i for i, s in enumerate(df["species"].unique())}
    y = np.array([species_map[s] for s in df["species"]], dtype=int)
    X = normalize(X)
    X_2d = PCA(n_components=2, random_state=random_state).fit_transform(X)
    X_2d = X_2d / (np.abs(X_2d).max(axis=0, keepdims=True) + 1e-12)
    return X_2d, y


def build_random_quadruplets_from_triplets(triplets, n_objects, q_count=50000, random_state=42):
    """Generate completely random quadruplets (for testing purposes)."""
    rng = np.random.default_rng(random_state)
    quads = []
    m = len(triplets)
    for _ in range(q_count):
        i, j, _ = triplets[rng.integers(m)]
        k, l = rng.choice(n_objects, size=2, replace=False)
        quads.append((int(i), int(j), int(k), int(l)))
    return np.asarray(quads, dtype=np.int64)