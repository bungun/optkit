import numpy as np
from ctypes import c_void_p, byref
from scipy.sparse import csc_matrix, csr_matrix

def gen_preconditioning_operator(dlib, olib, A_py, rho):
	n = A_py.shape[1]
	p_vec = dlib.vector(0, 0, None)
	dlib.vector_calloc(p_vec, n)
	p_ = np.zeros(n).astype(dlib.pyfloat)
	p_ptr = p_.ctypes.data_as(dlib.ok_float_p)

	# calculate diagonal preconditioner
	for j in xrange(A_py.shape[1]):
		p_[j] = 1. / (rho +  np.linalg.norm(A_py[:, j])**2)

	dlib.vector_memcpy_va(p_vec, p_ptr, 1)
	p = olib.diagonal_operator_alloc(p_vec)
	return p_, p_vec, p

def release_preconditioning_operator(dlib, olib, p_vec, p):
	olib.operator_free(p)
	dlib.vector_free(p_vec)

def gen_dense_operator(dlib, olib, A_py, rowmajor=True):
	m, n = A_py.shape
	order = dlib.enums.CblasRowMajor if rowmajor else \
			dlib.enums.CblasColMajor
	pyorder = 'C' if rowmajor else 'F'
	A_ = np.zeros(A_py.shape, order=pyorder).astype(dlib.pyfloat)
	A_ += A_py
	A_ptr = A_.ctypes.data_as(dlib.ok_float_p)
	A = dlib.matrix(0, 0, 0, None, order)
	dlib.matrix_calloc(A, m, n, order)
	dlib.matrix_memcpy_ma(A, A_ptr, order)
	o = olib.dense_operator_alloc(A)
	return A, o

def release_dense_operator(dlib, olib, A, o):
	olib.operator_free(o)
	dlib.matrix_free(A)

def gen_sparse_operator(dlib, slib, olib, A_py, rowmajor=True):
	m, n = A_py.shape
	order = dlib.enums.CblasRowMajor if rowmajor else \
			dlib.enums.CblasColMajor
	sparsemat = csr_matrix if rowmajor else csc_matrix
	sparse_hdl = c_void_p()
	slib.sp_make_handle(byref(sparse_hdl))
	A_ = A_py.astype(dlib.pyfloat)
	A_sp = csr_matrix(A_)
	A_ptr = A_sp.indptr.ctypes.data_as(slib.ok_int_p)
	A_ind = A_sp.indices.ctypes.data_as(slib.ok_int_p)
	A_val = A_sp.data.ctypes.data_as(dlib.ok_float_p)
	A = slib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
	slib.sp_matrix_calloc(A, m, n, A_sp.nnz, order)
	slib.sp_matrix_memcpy_ma(sparse_hdl, A, A_val, A_ind, A_ptr)
	slib.sp_destroy_handle(sparse_hdl)
	o = olib.sparse_operator_alloc(A)
	return A, o

def release_sparse_operator(slib, olib, A, o):
	olib.operator_free(o)
	slib.sp_matrix_free(A)
