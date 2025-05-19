import numpy as np
np.set_printoptions(suppress = True, linewidth=250, precision=4)


def inner_product(v: np.ndarray, w: np.ndarray) -> complex:
    r"""
    Computes the inner product of the two complex vectors v and w.
    If v and w are Hilbert space vectors it computes <v|w>.
    If they are matrices it computes Tr(v^H w).
    
    Arguments:
        v (np.array(complex)): complex vector
        w (np.array(complex)): complex vector
    
    Returns:
        (complex): the inner product
    """
    if v.shape != w.shape:
        raise ValueError("Vectors must have the same shape")
    return np.sum(np.conj(v) * w)


def norm(v: np.ndarray) -> float:
    r"""
    Computes the norm of a complex vector v.
    """
    return np.sqrt(inner_product(v, v))


def normalize(v: np.ndarray) -> np.ndarray:
    r"""
    Normalizes a complex vector v.
    """
    return v / norm(v)
