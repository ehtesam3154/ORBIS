"""Geometry helpers shared outside the scorer."""
import numpy as np


def kabsch(Y, X):
    """Y rigidly aligned onto X: rotation (+ reflection) + translation, no scale."""
    mY, mX = Y.mean(0), X.mean(0)
    U, _, Vt = np.linalg.svd((Y - mY).T @ (X - mX))
    return (Y - mY) @ (U @ Vt) + mX
