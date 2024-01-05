# distutils: language = c++
# distutils: sources = pronelib.cpp

import numpy as np
import ctypes
cimport numpy as np

cdef extern from "pronelib.hpp":
    cdef cppclass ProneKernel:
        ProneKernel()
        void run(double *projected_data, size_t n, size_t k, int *centers, int *assignment)

def prone(np.ndarray[np.double_t, ndim=2] X, int k):
    if k == 0:
        return [], []

    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]

    assert n > 0
    assert d > 0

    # first we sample a vector and perform the vector multiplication
    cdef np.ndarray[np.double_t, ndim=1] u = np.random.normal(size = (d, ))
    cdef np.ndarray[np.double_t, ndim=1] x = np.ascontiguousarray(X @ u)

    cdef np.ndarray[int, ndim=1, mode="c"] centers = np.ascontiguousarray(np.zeros((k, )), dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] assignment = np.ascontiguousarray(np.zeros_like(x), dtype=ctypes.c_int)

    cdef ProneKernel prone_kernel
    # prone_kernel = ProneKernel()
    prone_kernel.run(&x[0], n, k, &centers[0], &assignment[0])

    return centers, assignment

