"""Stage 4: cluster face embeddings into recurring identities (HDBSCAN)."""
import hdbscan
import numpy as np

import config


def l2norm(embs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(embs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return embs / n


def cluster_embeddings(embs, min_cluster_size=config.MIN_CLUSTER_SIZE,
                       min_samples=config.CLUSTER_MIN_SAMPLES,
                       epsilon=config.CLUSTER_SELECTION_EPSILON,
                       pca_dims=getattr(config, "CLUSTER_PCA_DIMS", 128)) -> np.ndarray:
    """Cluster face embeddings. For large high-dim inputs (real 512-d ArcFace), PCA-reduce
    before HDBSCAN — 512-d HDBSCAN is impractical at 100k+ faces, and ArcFace's intrinsic
    dimensionality is far below 512, so PCA-128 preserves cluster structure while making it
    tractable. Reference matching downstream still uses the FULL 512-d embeddings."""
    x = l2norm(np.asarray(embs, dtype=np.float64))
    if pca_dims and x.shape[1] > pca_dims and x.shape[0] > pca_dims:
        from sklearn.decomposition import PCA
        x = l2norm(PCA(n_components=int(pca_dims)).fit_transform(x))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_epsilon=float(epsilon),
        core_dist_n_jobs=-1,
    )
    return clusterer.fit_predict(x)


def centroids(embs_normed, labels) -> dict:
    out = {}
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        members = embs_normed[labels == lab]
        c = members.mean(axis=0)
        c /= (np.linalg.norm(c) or 1.0)
        out[int(lab)] = c
    return out


def cohesion(embs_normed, labels, cents) -> np.ndarray:
    out = np.full(len(labels), np.nan)
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        out[i] = float(embs_normed[i] @ cents[int(lab)])
    return out


def merge_clusters(labels, embs_normed, threshold=config.CLUSTER_MERGE_THRESHOLD) -> np.ndarray:
    """Stage 4b: merge clusters whose centroids are cosine >= threshold (same person,
    different shots) into single identities via single-linkage union-find. Noise (-1) stays
    noise. Returns relabelled (contiguous from 0) labels aligned to `labels`."""
    labels = np.asarray(labels)
    labs = sorted(int(l) for l in set(labels.tolist()) if l != -1)
    if len(labs) <= 1:
        return labels.copy()
    cents = centroids(embs_normed, labels)
    M = np.vstack([cents[l] for l in labs])           # (k, d), unit vectors
    sim = M @ M.T                                      # pairwise centroid cosine
    parent = list(range(len(labs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            if sim[i, j] >= threshold:
                parent[find(j)] = find(i)             # union

    root_to_new, nid = {}, 0
    for i in range(len(labs)):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = nid
            nid += 1
    old_to_new = {labs[i]: root_to_new[find(i)] for i in range(len(labs))}
    return np.array([-1 if l == -1 else old_to_new[int(l)] for l in labels])
