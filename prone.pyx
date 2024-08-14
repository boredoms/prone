# distutils: language = c++
# distutils: sources = pronelib.cpp

import numpy as np
import ctypes
cimport numpy as np

cdef extern from "pronelib.hpp":
    cdef cppclass ProneKernel:
        ProneKernel()
        void run(double *projected_data, int n, int k, int *centers, int *assignment)
        void coreset(double *dataset, int n, int d, int *centers, int k, int *assignment, int coreset_size, int *coreset_indices, double *coreset_weights)

# Main function to run the prone algorithm for creating a clustering and assignment
def prone(np.ndarray[np.double_t, ndim=2] X, int k):
    if k == 0:
        return [], []

    if X.shape[0] == 0 or X.shape[1] == 0:
        print("Error: Dataset is empty")
        return [], []

    # first we sample a vector and perform the vector multiplication
    cdef np.ndarray[np.double_t, ndim=1] u = np.random.normal(size = (X.shape[1], ))
    cdef np.ndarray[np.double_t, ndim=1] x = np.ascontiguousarray(X @ u)

    cdef np.ndarray[int, ndim=1, mode="c"] centers = np.ascontiguousarray(np.zeros((k, )), dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] assignment = np.ascontiguousarray(np.zeros_like(x), dtype=ctypes.c_int)

    cdef ProneKernel prone_kernel
    # prone_kernel = ProneKernel()
    prone_kernel.run(&x[0], X.shape[0], k, &centers[0], &assignment[0])

    return centers, assignment

# Fast coreset function which uses Prone as the bicriteria solution.
def coreset(np.ndarray[np.double_t, ndim=2] X, int k, int coreset_size):
    if k == 0 or coreset_size == 0:
        return [], []
    
    centers, assignment = prone(X, k)

    cdef np.ndarray[int, ndim=1, mode="c"] cs = np.ascontiguousarray(centers, dtype=ctypes.c_int)
    cdef np.ndarray[int, ndim=1, mode="c"] ass = np.ascontiguousarray(assignment, dtype=ctypes.c_int)

    X = np.ascontiguousarray(X)

    cdef np.ndarray[int, ndim=1, mode="c"] coreset_indices = np.ascontiguousarray(np.zeros((coreset_size, )), dtype=ctypes.c_int)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] coreset_weights = np.ascontiguousarray(np.zeros((coreset_size, )), dtype=ctypes.c_double)
    
    cdef ProneKernel prone_kernel
    prone_kernel.coreset(&X[0, 0], X.shape[0], X.shape[1], &cs[0], k, &ass[0], coreset_size, &coreset_indices[0], &coreset_weights[0])

    return coreset_indices, coreset_weights