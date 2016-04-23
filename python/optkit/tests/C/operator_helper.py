import numpy as np
from ctypes import c_void_p, byref
from scipy.sparse import csc_matrix, csr_matrix

def gen_dense_operator(lib, A_py, rowmajor=True):
	m, n = A_py.shape
	order = lib.enums.CblasRowMajor if rowmajor else \
			lib.enums.CblasColMajor
	pyorder = 'C' if rowmajor else 'F'
	A_ = np.zeros(A_py.shape, order=pyorder).astype(lib.pyfloat)
	A_ += A_py
	A_ptr = A_.ctypes.data_as(lib.ok_float_p)
	A = lib.matrix(0, 0, 0, None, order)
	lib.matrix_calloc(A, m, n, order)
	lib.matrix_memcpy_ma(A, A_ptr, order)
	o = lib.dense_operator_alloc(A)
	return A, o, lib.matrix_free

def gen_sparse_operator(lib, A_py, rowmajor=True):
	m, n = A_py.shape
	order = lib.enums.CblasRowMajor if rowmajor else \
			lib.enums.CblasColMajor
	sparsemat = csr_matrix if rowmajor else csc_matrix
	sparse_hdl = c_void_p()
	lib.sp_make_handle(byref(sparse_hdl))
	A_ = A_py.astype(lib.pyfloat)
	A_sp = csr_matrix(A_)
	A_ptr = A_sp.indptr.ctypes.data_as(lib.ok_int_p)
	A_ind = A_sp.indices.ctypes.data_as(lib.ok_int_p)
	A_val = A_sp.data.ctypes.data_as(lib.ok_float_p)
	A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
	lib.sp_matrix_calloc(A, m, n, A_sp.nnz, order)
	lib.sp_matrix_memcpy_ma(sparse_hdl, A, A_val, A_ind, A_ptr)
	lib.sp_destroy_handle(sparse_hdl)
	o = lib.sparse_operator_alloc(A)
	return A, o, lib.sp_matrix_free