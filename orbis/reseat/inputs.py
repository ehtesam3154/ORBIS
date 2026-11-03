"""Engine inputs of the reseating step: the sample (sites), the one-call best guess fitted rigidly onto it, and the reader arrays.

Layout files are npz with keys P (model units), G (the test cells' prebuilt positions, an identity check), cc (cell classes, an identity
check) and iso (true-scale flag). Positions are converted to micrometres as P x mm_per_unit x 1000; the best guess is aligned onto the
sample by a rigid fit (rotation, reflection and translation, no scale)."""
import hashlib

import numpy as np

MIN_CELLS = 40


def md5(path):
    """md5 of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def kabsch(Y, X):
    """Y rigidly aligned onto X: rotation (reflection allowed) and translation, no scale."""
    mY, mX = Y.mean(0), X.mean(0)
    U, _, Vt = np.linalg.svd((Y - mY).T @ (X - mX))
    return (Y - mY) @ (U @ Vt) + mX


def as_coords(A, name, N=None, D=None):
    """Validated float64 coordinates (N, D); raises ValueError otherwise."""
    A = np.asarray(A)
    if A.dtype.kind not in "fiu":
        raise ValueError(f"{name}: dtype {A.dtype} is not numeric")
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] not in (2, 3):
        raise ValueError(f"{name}: shape {A.shape}; expected (N, 2) or (N, 3)")
    if N is not None and A.shape[0] != N:
        raise ValueError(f"{name}: {A.shape[0]} cells, the test set has {N}")
    if D is not None and A.shape[1] != D:
        raise ValueError(f"{name}: {A.shape[1]}-D coordinates, the test set is {D}-D")
    bad = ~np.isfinite(A).all(1)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} cells with NaN / inf coordinates")
    if A.shape[0] < MIN_CELLS:
        raise ValueError(f"{name}: only {A.shape[0]} cells")
    return A


def load_layout(path, pos, cell_class):
    """Layout npz -> P (float64, model units) after the identity checks against the test set's pos.npy and cell_class.npy."""
    z = np.load(path)
    for k in ("P", "G", "cc", "iso"):
        if k not in z.files:
            raise ValueError(f"{path}: npz lacks '{k}'")
    if not bool(z["iso"]):
        raise ValueError(f"{path}: not a true-scale (iso) layout")
    P = as_coords(z["P"], path, N=len(pos), D=pos.shape[1])
    if not np.array_equal(z["G"], pos):
        raise ValueError(f"{path}: G differs from the test set's pos.npy (different cells or order)")
    if not np.array_equal(np.asarray(z["cc"]).astype(np.int64), cell_class):
        raise ValueError(f"{path}: cc differs from the test set's cell_class.npy (different cells or order)")
    return P


def build(sample, guess, reader, test_dir, mm_per_unit, reader_md5=None):
    """S (sites, um), G (best guess fitted onto S, um), p, q = (q32, q128), isig = (1 / sigma32, 1 / sigma128), the sample npz and md5s."""
    pos = np.load(f"{test_dir}/pos.npy")
    cc = np.load(f"{test_dir}/cell_class.npy").astype(np.int64)
    P0 = load_layout(sample, pos, cc)
    z0 = np.load(sample)
    S = P0.astype(np.float64) * mm_per_unit * 1000.0
    G = kabsch(load_layout(guess, pos, cc).astype(np.float64) * mm_per_unit * 1000.0, S)
    rmd5 = md5(reader)
    if reader_md5 and rmd5 != reader_md5:
        raise ValueError(f"{reader}: md5 {rmd5} is not the expected {reader_md5}")
    R = np.load(reader)
    p = R["p"].astype(np.float64)
    q = (R["q32"].astype(np.float64), R["q128"].astype(np.float64))
    isig = (1.0 / R["sigma32"].astype(np.float64), 1.0 / R["sigma128"].astype(np.float64))
    assert len(p) == len(S) == len(q[0])
    return dict(S=S, p=p, G=G, q=q, isig=isig, z0=z0, sample_md5=md5(sample), guess_md5=md5(guess), reader_md5=rmd5)
