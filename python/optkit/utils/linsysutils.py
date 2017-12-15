# from optkit.compat import *

# import numpy as np
# import scipy.sparse as sp
# import ctypes as ct
# import toolz

# # low-level utilities
# @toolz.curry
# def util_make_cvector(denselib, x):
#   if x is None:
#       return denselib.vector(0, 0, None)
#   elif isinstance(x, np.ndarray):
#       if len(x.shape) == 1:
#           x_ = denselib.vector(0, 0, None)
#           denselib.vector_calloc(x_, x.size)
#           denselib.vector_memcpy_va(x_,
#                                     x.ctypes.data_as(denselib.ok_float_p), 1)
#       else:
#           raise ValueError('argument "x", if provided as an {}, must be a'
#                            'vector, i.e., singleton dimension. dimensions '
#                            'of provided array: {}'.format(np.ndarray, x.shape))
#   else:
#       raise TypeError('argument "x" must be of type {} or {}'.format(
#                       type(None), np.ndarray))




# class UtilMakeCMatrix(object):
#   def __init__(self, denselib):
#       self.denselib = denselib
#       self.enums = denselib.enums

#   def __call__(self, A=None, copy_data=True):

#       if A is None:
#           return self.denselib.matrix(0, 0, 0, None,
#               self.enums.CblasRowMajor)
#       elif isinstance(A, np.ndarray) and len(A.shape)==2:
#           (m,n) = A.shape

#           order = self.enums.CblasRowMajor if A.flags.c_contiguous else \
#                   self.enums.CblasColMajor
#           pytype = np.float32 if denselib.FLOAT else np.float64

#           A_ = self.denselib.matrix(0, 0, 0, None, order)
#           if not copy_data:
#               self.denselib.matrix_view_array(A_, A.ctypes.data_as(
#                   self.denselib.ok_float_p), m, n, order)
#           else:
#               self.denselib.matrix_calloc(A_, m, n, order)
#               self.denselib.matrix_memcpy_ma(A_, A.ctypes.data_as(
#                   self.denselib.ok_float_p), order)
#           return A_
#       else:
#           return None
#           # TODO: error message (type, dims)


# class UtilMakeCSparseMatrix(object):
#   def __init__(self, denselib, sparselib):
#       self.sparselib = sparselib
#       self.enums = denselib.enums

#   def __call__(self, A=None):
#       if A is None:
#           return self.sparselib.sparse_matrix(0, 0, 0, 0,
#               None, None, None, self.enums.CblasRowMajor)
#       elif isinstance(A, (sp.csr_matrix, sp.csc_matrix)):
#           (m,n) = A.shape
#           order = self.enums.CblasRowMajor if isinstance(A, sp.csr_matrix) else \
#                   self.enums.CblasColMajor

#           A_ = self.sparselib.sparse_matrix(0, 0, 0, 0,
#               None, None, None, order)

#           sparse_handle = ct.c_void_p()
#           self.sparselib.sp_make_handle(ct.byref(sparse_handle))
#           self.sparselib.sp_matrix_calloc(A_, m, n, A.nnz, order)
#           self.sparselib.sp_matrix_memcpy_ma(sparse_handle, A_,
#               A.data.ctypes.data_as(self.sparselib.ok_int_p),
#               A.indices.ctypes.data_as(self.sparselib.ok_int_p),
#               A.indptr.ctypes.data_as(self.sparselib.ok_float_p),
#               order)
#           self.sparselib.sp_destroy_handle(sparse_handle)
#           return A_
#       else:
#           return None
#           # TODO: error message (type, dims)


# @toolz.curry
# def util_release_cvector(denselib, x):
#   if isinstance(x, denselib.vector):
#       denselib.vector_free(x)

# @toolz.curry
# def util_release_cmatrix(denselib, A):
#   if isinstance(A, denselib.matrix):
#       denselib.matrix_free(A)

# @toolz.curry
# def util_release_csparsematrix(sparselib, A):
#   if isinstance(A, sparselib.sparse_matrix):
#       sparselib.sp_matrix_free(A)