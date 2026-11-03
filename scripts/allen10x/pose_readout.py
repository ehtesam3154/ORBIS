"""Label-free pose readout: a placement is put into the training frame by one orthogonal transform (rotation or reflection, no
scale) and a translation, fitted on the centres of 64 expression clusters (k-means on the training cells' features).
No position, label, region, donor or section of the placed cells enters any function here."""
import json
import time

import numpy as np

N_CLUSTERS = 64
MAX_TRAIN_CELLS = 1_000_000
SEED = 25000
MIN_QUERY_PER_CLUSTER = 50
MIN_SUPPORTED_CLUSTERS = 6
MIN_SV_RATIO = 0.01
ASSIGN_CHUNK = 200_000


def draw_sample(n_total, max_cells=MAX_TRAIN_CELLS, seed=SEED):
    if n_total <= max_cells:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, size=max_cells, replace=False))


def fit_landmarks(train_feat, train_pos, col_order=None, k=N_CLUSTERS, seed=SEED, max_cells=MAX_TRAIN_CELLS, verbose=0):
    from sklearn.cluster import KMeans
    import sklearn
    t0 = time.time()
    idx = draw_sample(len(train_feat), max_cells, seed)
    X = np.asarray(train_feat[idx], dtype=np.float32)
    if col_order is not None:
        X = np.ascontiguousarray(X[:, col_order])
    pos = np.asarray(train_pos[idx], dtype=np.float64)
    km = KMeans(n_clusters=k, random_state=seed, verbose=verbose)
    lab = km.fit_predict(X)
    n_s = np.bincount(lab, minlength=k)
    if (n_s == 0).any():
        raise RuntimeError(f"empty k-means cluster(s): {np.flatnonzero(n_s == 0).tolist()}")
    landmark = np.stack([np.bincount(lab, weights=pos[:, a], minlength=k) / n_s for a in range(3)], 1)
    info = dict(sklearn=sklearn.__version__, params={p: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                      for p, v in km.get_params().items()},
                n_iter=int(km.n_iter_), inertia=float(km.inertia_), n_cells_fitted=int(len(idx)), n_train_total=int(len(train_feat)),
                sample_seed=int(seed), seconds=round(time.time() - t0, 1))
    return dict(centers=km.cluster_centers_.astype(np.float32), landmark=landmark, n_sample=n_s, sample_idx=idx, kmeans=info)


def assign(feat, centers, chunk=ASSIGN_CHUNK, col_order=None):
    C = np.asarray(centers, dtype=np.float64)
    c2 = (C ** 2).sum(1)
    out = np.empty(len(feat), dtype=np.int16)
    for s in range(0, len(feat), chunk):
        x = np.asarray(feat[s:s + chunk], dtype=np.float32)
        if col_order is not None:
            x = x[:, col_order]
        out[s:s + chunk] = np.argmin(c2[None, :] - 2.0 * (x.astype(np.float64) @ C.T), axis=1)
    return out


def weighted_cov(L, w):
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    m = w @ L
    D = L - m
    return (D * w[:, None]).T @ D


def sv_ratio(M):
    s = np.linalg.svd(M, compute_uv=False)
    return float(s.min() / s.max())


def procrustes(pred_land, target_land, w):
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mp, mt = w @ pred_land, w @ target_land
    H = (pred_land - mp).T @ ((target_land - mt) * w[:, None])
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    t = mt - mp @ R
    return R, t, S


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R * np.sign(np.linalg.det(R))) - 1) / 2, -1, 1))))


def fit_pose(landmarks, labels, pred, scale=None):
    k = len(landmarks["landmark"])
    scale = np.ones(3) if scale is None else np.asarray(scale, dtype=np.float64)
    n_q = np.bincount(labels, minlength=k)
    sup = n_q >= MIN_QUERY_PER_CLUSTER
    pred = np.asarray(pred, dtype=np.float64)
    pl = np.stack([np.bincount(labels, weights=pred[:, a], minlength=k) / np.maximum(n_q, 1) for a in range(3)], 1)
    w = n_q[sup].astype(np.float64)
    c = (w / w.sum()) @ landmarks["landmark"][sup]
    tgt = (landmarks["landmark"][sup] - c) / scale
    R, t, S = procrustes(pl[sup], tgt, w)
    cov_t, cov_p = weighted_cov(tgt, w), weighted_cov(pl[sup], w)
    resid = np.linalg.norm(pl[sup] @ R + t - tgt, axis=1)
    return dict(R=R, t=t, c=c, scale=scale, n_query=n_q, supported=sup, pred_landmark=pl, target_landmark=tgt, weights=w,
                n_supported=int(sup.sum()), n_supported_cells=int(n_q[sup].sum()), det=float(np.linalg.det(R)),
                sv_ratio_target_cov=sv_ratio(cov_t), sv_ratio_pred_cov=sv_ratio(cov_p), sv_ratio_cross=float(S.min() / S.max()),
                sv_ratio_target_cov_unweighted=sv_ratio(weighted_cov(tgt, np.ones(len(tgt)))),
                cross_sv=S.tolist(), cov_target_sv=np.linalg.svd(cov_t, compute_uv=False).tolist(),
                resid_rms=float(np.sqrt(np.average(resid ** 2, weights=w))), angle_deg=rotation_angle_deg(R))


def conditions(pose, ref_det_sign=None):
    out = dict(supported_ge6=bool(pose["n_supported"] >= MIN_SUPPORTED_CLUSTERS),
               sv_ratio_ge_0p01=bool(pose["sv_ratio_target_cov"] >= MIN_SV_RATIO))
    if ref_det_sign is not None:
        out["reflection_sign_equals_reference"] = bool(np.sign(pose["det"]) == np.sign(ref_det_sign))
    out["all"] = bool(all(out.values()))
    return out


def apply_pose(P, pose):
    return (np.asarray(P, dtype=np.float64) @ pose["R"] + pose["t"]) * pose["scale"] + pose["c"]


def mean_graph_extent(pos, graph_ids):
    import pandas as pd
    gid = pd.factorize(np.asarray(graph_ids))[0]
    pos = np.asarray(pos, dtype=np.float64)
    o = np.argsort(gid, kind="stable")
    cut = np.flatnonzero(np.diff(gid[o])) + 1
    ext = np.array([p.max(0) - p.min(0) for p in np.split(pos[o], cut)])
    return ext.mean(0), ext


def save_landmarks(path, lm):
    np.savez_compressed(path, centers=lm["centers"], landmark=lm["landmark"], n_sample=lm["n_sample"], sample_idx=lm["sample_idx"],
                        kmeans_json=np.array(json.dumps(lm["kmeans"])))


def load_landmarks(path):
    z = np.load(path)
    return dict(centers=z["centers"], landmark=z["landmark"], n_sample=z["n_sample"], sample_idx=z["sample_idx"],
                kmeans=json.loads(str(z["kmeans_json"])))
