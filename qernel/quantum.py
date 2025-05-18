import numpy as np
np.set_printoptions(suppress = True, linewidth=250, precision=4)

def inner_product(v, w):
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
    return np.sum(np.conj(v) * w)