"""Arnoldi algorithm."""
import numpy as np
from scipy.linalg import lu, solve_triangular


def _mgs_step(v, q):
    """Single step of a modified Gram--Schmidt algorithm.

    Parameters
    ----------
    v : (n,) ndarray
        current vector

    q : (n, m) ndarray
        previous vector directions

    Returns
    -------
    (n,) ndarray
        normalised vector orthogonal to previous directions

    """
    for j in range(q.shape[1]):
        v -= np.dot(q[:, j], v) * q[:, j]
    return v / np.linalg.norm(v)


def arnoldi(a, b, num_directions):
    """Perform the Arnoldi algorithm.

    Parameters
    ----------
    a : (n, n) ndarray
        matrix A

    b : (n,) ndarray
        vector b

    num_directions : int
        number of directions in the rational Krylov subspace

    Returns
    -------
    (n, num_directions) ndarray
        rational Krylov subspace, i.e.
        span(rks) = span([b, A^-1 b, A^-2 b, ...])

    """
    rks = np.empty((a.shape[0], num_directions))
    p, l, u = lu(a)
    rks[:, 0] = b.squeeze() / np.linalg.norm(b.squeeze())
    for i in range(1, num_directions):
        x = solve_triangular(l, p.T @ rks[:, i - 1], lower=True)
        x = solve_triangular(u, x, lower=False)
        rks[:, i] = _mgs_step(x, rks[:, :i])
    return rks
