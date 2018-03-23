"""Test the Arnoldi implementation."""
from dla.linalg import arnoldi
import numpy as np
import numpy.testing as npt
from scipy.linalg import subspace_angles


a = np.array([
    [9.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 8.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 5.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 6.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 8.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 5.0],
    ])
b = np.array([
    [1.0],
    [0.0],
    [3.0],
    [1.0],
    [0.0],
    [0.0],
    ])


def test_arnoldi_simple():
    """Test the Arnoldi algorithm for a simple system."""
    num_directions = 3
    x = np.empty((a.shape[0], num_directions))
    x[:, 0] = b.squeeze()
    x[:, 0] /= np.linalg.norm(x[:, 0])
    for i in range(1, num_directions):
        x[:, i] = np.linalg.solve(a, x[:, i - 1])
        x[:, i] /= np.linalg.norm(x[:, i])
    rks = arnoldi(a, b, num_directions)
    npt.assert_almost_equal(np.abs(subspace_angles(x, rks)).max(), 0.0)


def test_arnoldi_xxl():
    """Test the Arnoldi algorithm for a larger system."""
    np.random.seed(777)
    a_xxl = np.random.rand(100, 100)
    b_xxl = np.random.rand(100)
    num_directions = 10

    x = np.empty((a_xxl.shape[0], num_directions))
    x[:, 0] = b_xxl
    x[:, 0] /= np.linalg.norm(x[:, 0])
    for i in range(1, num_directions):
        x[:, i] = np.linalg.solve(a_xxl, x[:, i - 1])
        x[:, i] /= np.linalg.norm(x[:, i])
    rks = arnoldi(a_xxl, b_xxl, num_directions)
    npt.assert_almost_equal(np.abs(subspace_angles(x, rks)).max(), 0.0)


def test_arnoldi_orthogonality():
    """Test if the Arnoldi implementation produces an orthogonal basis."""
    num_directions = 4
    rks = arnoldi(a, b, num_directions)
    for i in range(num_directions):
        for j in range(i):
            npt.assert_almost_equal(np.dot(rks[:, i], rks[:, j]), 0.0)


def test_arnoldi_normalisation():
    """Test if the Arnoldi implementation produces an normalised basis."""
    num_directions = 4
    rks = arnoldi(a, b, num_directions)
    npt.assert_almost_equal(
        np.linalg.norm(rks, axis=0),
        np.ones((num_directions,)),
        )
