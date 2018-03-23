"""Helper functions for linear algebra."""
import numpy as np


def modified_gram_schmidt(v):
    """Orthonormalise vectors using the modified Gram--Schmidt process.

    Parameters
    ----------
    v : (n, m) ndarray
        horizontally stacked vectors of dimension n. It is assumed that m <= n.

    Returns
    -------
    (n, m) ndarray
        orthonormalised vectors such that span(w) = span(v)

    """
    w = np.empty(v.shape)
    for k in range(v.shape[1]):
        w[:, k] = v[:, k]
        for j in range(k):
            w[:, k] -= np.dot(w[:, j], v[:, k]) * w[:, j]
        w[:, k] /= np.linalg.norm(w[:, k])
    return w


def projector_onto_kernel(a):
    """Compute the orthogonal projection matrix P onto the kernel of A.

    This function uses an SVD-based method described in Golub.

    If all rows of A are linearly independent,
    you can probably use a simpler/faster method.

    Parameters
    ----------
    a : (n, m) ndarray
        input matrix A. It is assumed that n <= m

    Returns
    -------
    (m, m) ndarray
        orthogonal projector onto the kernel of A

    """
    r = np.linalg.matrix_rank(a)
    _, _, v = np.linalg.svd(a)
    v_tilde = v.T[:, r:]
    return v_tilde @ v_tilde.T


def online_variance(iterator, item_shape):
    """Compute the elementwise variance in one pass using an online algo.

    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Parameters
    ----------
    iterator : iterator
        an iterator yielding ndarrays of the same shape

    item_shape : tuple of ints
        a shape specification as a tuple

    Returns
    -------
    ndarray
        elementwise variance of the items in the iterator

    """
    n = 0
    mean = np.zeros(item_shape)
    m2 = np.zeros(item_shape)
    for item in iterator:
        n += 1
        delta = item - mean
        mean += delta/n
        delta2 = item - mean
        m2 += delta*delta2

    if n < 2:
        return float('nan')
    else:
        return m2 / n


def log_rel_error(testee, reference):
    """Compute the log relative error of a vector to some reference.

    Be careful if reference is near zero!

    Parameters
    ----------
    testee : (n,) ndarray
        an inexact result

    reference : (n,) ndarray
        an exact reference

    Returns
    -------
    float
        log relative error, i.e. log10(||testee - reference|| / ||reference||)

    """
    res = np.log10(np.linalg.norm(testee - reference))
    res -= np.log10(np.linalg.norm(reference))
    return res
