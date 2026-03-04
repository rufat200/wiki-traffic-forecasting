# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np


cpdef rolling_slope(const float[:] x, int window):
    cdef int n = x.shape[0]
    cdef int i, j
    cdef float x_mean, num, s
    cdef np.ndarray[np.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)
    cdef float t_mean = (window - 1) / 2.0
    cdef float t_var = 0.0
    for j in range(window):
        t_var += (j - t_mean) * (j - t_mean)
    for i in range(window-1):
        out[i] = np.nan
    with nogil:
        for i in range(window-1, n):
            s = 0.0
            for j in range(window):
                s += x[i - window + 1 + j]
            x_mean = s / window
            num = 0.0
            for j in range(window):
                num += (j - t_mean) * (x[i - window + 1 + j] - x_mean)
            out[i] = num / t_var
    return out
