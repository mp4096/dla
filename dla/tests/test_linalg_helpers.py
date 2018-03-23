"""Tests the linear algebra helper functions."""
from dla.linalg import (
    log_rel_error,
    modified_gram_schmidt,
    online_variance,
    projector_onto_kernel,
    )
import numpy as np
import numpy.testing as npt
import scipy as sp


def test_mgs_already_orthonormal():
    """Test MGS if vectors are already orthonormal."""
    v = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        ])
    w = modified_gram_schmidt(v)
    assert np.all(v == w)


def test_mgs_normalisation():
    """Test if MGS normalises correctly."""
    v = np.array([
        [1.0, 1.0, 3.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 4.0],
        ])
    w = modified_gram_schmidt(v)
    npt.assert_almost_equal(
        np.linalg.norm(w, axis=0),
        np.ones((v.shape[1],)),
        )


def test_mgs_copy():
    """Test if MGS preserves the original matrix."""
    v = np.array([
        [1.0, 1.0, 3.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 4.0],
        ])
    v_copy = v.copy()
    _ = modified_gram_schmidt(v)
    assert np.all(v_copy == v)


def test_mgs_orthogonality():
    """Test if MGS orthogonalises the directions."""
    v = np.array([
        [1.0, 1.0, 3.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 4.0],
        ])
    w = modified_gram_schmidt(v)
    for i in range(v.shape[1]):
        for j in range(i):
            npt.assert_almost_equal(np.dot(w[:, i], w[:, j]), 0.0)


def test_mgs_orthogonality_xxl():
    """Test if MGS orthogonalises the directions -- large vectors."""
    np.random.seed(777)
    v = np.random.randn(333, 33)
    w = modified_gram_schmidt(v)
    for i in range(v.shape[1]):
        for j in range(i):
            npt.assert_almost_equal(np.dot(w[:, i], w[:, j]), 0.0)


def test_mgs_vs_qr():
    """Test if MGS result spans the same subspace as the QR decomposition."""
    np.random.seed(777)
    v = np.random.randn(100, 20)
    w = modified_gram_schmidt(v)
    q, _ = np.linalg.qr(v)
    npt.assert_almost_equal(
        sp.linalg.subspace_angles(q, w),
        np.zeros((v.shape[1],)),
        )


def test_kernel_proj():
    """Test the projector onto the kernel of A."""
    np.random.seed(777)
    n, m = 200, 20
    a = np.random.randn(m, n)
    p = projector_onto_kernel(a)
    assert p.shape == (n, n)
    x = np.random.randn(n)
    npt.assert_almost_equal(
        a @ (p @ x),
        np.zeros((m,)),
        )


def test_online_variance_scalar():
    """Test the online variance algorithm, scalar items."""
    np.random.seed(777)
    item_shape = (1,)
    iterator = [np.random.rand(1) for _ in range(10)]
    online = online_variance(iterator, item_shape)
    offline = np.var(iterator)
    npt.assert_almost_equal(online[0], offline)


def test_online_variance_matrix():
    """Test the online variance algorithm, scalar items."""
    np.random.seed(777)
    item_shape = (100, 200)
    iterator = [np.random.rand(*item_shape) for _ in range(10)]
    online = online_variance(iterator, item_shape)
    offline = np.var(iterator, axis=0)
    npt.assert_almost_equal(online, offline)


def test_log_rel_error_simple():
    """Test the log relative error implementation."""
    size = 100
    testee = np.ones((size,)) + 1.0e-3
    reference = np.ones((size,))
    npt.assert_almost_equal(log_rel_error(testee, reference), -3.0)
