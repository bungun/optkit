from optkit.compat import *

import numpy as np
import ctypes as ct

from optkit.libs.loader import OptkitLibs

def include_ok_defs(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'ok_float', attach_base_ctypes, **include_args)
def include_ok_dense(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'vector_p', attach_dense_linsys_ctypes, **include_args)
def include_ok_sparse(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'sparse_matrix_p', attach_sparse_linsys_ctypes, **include_args)

def ok_base_API(): return [attach_base_ccalls]
def ok_vector_API(): return ok_base_API() + [attach_vector_ccalls]
def ok_dense_API(): return ok_vector_API() + [attach_dense_linsys_ccalls]
def ok_sparse_API(): return ok_vector_API() + [attach_sparse_linsys_ccalls]
def ok_linsys_API(): return ok_dense_API() + [attach_sparse_linsys_ccalls]

class DenseLinsysLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libok_dense_', ok_dense_API())

class SparseLinsysLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libok_sparse_', ok_sparse_API())

def attach_base_ctypes(lib, single_precision=False):
	# low-level C types, as defined for optkit
	lib.ok_float = ct.c_float if single_precision else ct.c_double
	lib.ok_int = ct.c_int
	lib.pyfloat = np.float32 if single_precision else np.float64

	# pointers to C types
	lib.c_int_p = ct.POINTER(ct.c_int)
	lib.c_size_t_p = ct.POINTER(ct.c_size_t)
	lib.ok_float_p = ct.POINTER(lib.ok_float)
	lib.ok_int_p = ct.POINTER(lib.ok_int)

	def ok_float_pointerize(numpy_array):
		return numpy_array.ctypes.data_as(lib.ok_float_p)
	def ok_int_pointerize(numpy_array):
		return numpy_array.ctypes.data_as(lib.ok_int_p)
	lib.ok_float_pointerize = ok_float_pointerize
	lib.ok_int_pointerize = ok_int_pointerize

def attach_dense_linsys_ctypes(lib, single_precision=False):
	include_ok_defs(lib, single_precision=single_precision)

	# vector struct
	class ok_vector(ct.Structure):
		_fields_ = [('size', ct.c_size_t),
					('stride', ct.c_size_t),
					('data', lib.ok_float_p)]
	lib.vector = ok_vector
	lib.vector_p = ct.POINTER(lib.vector)

	# index vector struct
	class ok_indvector(ct.Structure):
		_fields_ = [('size', ct.c_size_t),
					('stride', ct.c_size_t),
					('data', lib.c_size_t_p)]
	lib.indvector = ok_indvector
	lib.indvector_p = ct.POINTER(lib.indvector)

	# integer vector struct
	class ok_int_vector(ct.Structure):
		_fields_ = [('size', ct.c_size_t),
					('stride', ct.c_size_t),
					('data', lib.c_int_p)]
	lib.int_vector = ok_int_vector
	lib.int_vector_p = ct.POINTER(lib.int_vector)

	# matrix struct
	class ok_matrix(ct.Structure):
		_fields_ = [('size1', ct.c_size_t),
					('size2', ct.c_size_t),
					('ld', ct.c_size_t),
					('data', lib.ok_float_p),
					('order', ct.c_uint)]
	lib.matrix = ok_matrix
	lib.matrix_p = ct.POINTER(lib.matrix)

def attach_sparse_linsys_ctypes(lib, single_precision=False):
	include_ok_defs(lib, single_precision=single_precision)

	# sparse matrix struct
	class ok_sparse_matrix(ct.Structure):
		_fields_ = [('size1', ct.c_size_t),
					('size2', ct.c_size_t),
					('nnz', ct.c_size_t),
					('ptrlen', ct.c_size_t),
					('val', lib.ok_float_p),
					('ind', lib.ok_int_p),
					('ptr', lib.ok_int_p),
					('order', ct.c_uint)]

	lib.sparse_matrix = ok_sparse_matrix
	lib.sparse_matrix_p = ct.POINTER(lib.sparse_matrix)

def attach_base_ccalls(lib, single_precision=False):
	include_ok_defs(lib, single_precision=single_precision)

	c_int_p = lib.c_int_p

	lib.optkit_version.argtypes = [c_int_p, c_int_p, c_int_p, c_int_p]
	lib.optkit_version.restype = ct.c_uint

	lib.ok_device_reset.argtypes = []
	lib.ok_device_reset.restype = ct.c_uint

def attach_vector_ccalls(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	c_int_p = lib.c_int_p
	c_size_t_p = lib.c_size_t_p
	vector_p = lib.vector_p
	indvector_p = lib.indvector_p
	int_vector_p = lib.int_vector_p

	lib.vector_alloc.argtypes = [vector_p, ct.c_size_t]
	lib.vector_calloc.argtypes = [vector_p, ct.c_size_t]
	lib.vector_free.argtypes = [vector_p]
	lib.vector_set_all.argtypes = [vector_p, ok_float]
	lib.vector_subvector.argtypes = [
			vector_p, vector_p, ct.c_size_t, ct.c_size_t]
	lib.vector_view_array.argtypes = [vector_p, ok_float_p, ct.c_size_t]
	lib.vector_memcpy_vv.argtypes = [vector_p, vector_p]
	lib.vector_memcpy_va.argtypes = [vector_p, ok_float_p, ct.c_size_t]
	lib.vector_memcpy_av.argtypes = [ok_float_p, vector_p, ct.c_size_t]
	lib.vector_print.argtypes = [vector_p]
	lib.vector_scale.argtypes = [vector_p, ok_float]
	lib.vector_add.argtypes = [vector_p, vector_p]
	lib.vector_sub.argtypes = [vector_p, vector_p]
	lib.vector_mul.argtypes = [vector_p, vector_p]
	lib.vector_div.argtypes = [vector_p, vector_p]
	lib.vector_add_constant.argtypes = [vector_p, ok_float]
	lib.vector_abs.argtypes = [vector_p]
	lib.vector_recip.argtypes = [vector_p]
	lib.vector_sqrt.argtypes = [vector_p]
	lib.vector_pow.argtypes = [vector_p, ok_float]
	lib.vector_exp.argtypes = [vector_p]
	lib.vector_indmin.argtypes = [vector_p, c_size_t_p]
	lib.vector_min.argtypes = [vector_p, ok_float_p]
	lib.vector_max.argtypes = [vector_p, ok_float_p]

	lib.indvector_alloc.argtypes = [indvector_p, ct.c_size_t]
	lib.indvector_calloc.argtypes = [indvector_p, ct.c_size_t]
	lib.indvector_free.argtypes = [indvector_p]
	lib.indvector_set_all.argtypes = [indvector_p, ok_float]
	lib.indvector_subvector.argtypes = [indvector_p, indvector_p,
										ct.c_size_t, ct.c_size_t]
	lib.indvector_view_array.argtypes = [indvector_p, c_size_t_p,
										 ct.c_size_t]
	lib.indvector_memcpy_vv.argtypes = [indvector_p, indvector_p]
	lib.indvector_memcpy_va.argtypes = [indvector_p, c_size_t_p,
										ct.c_size_t]
	lib.indvector_memcpy_av.argtypes = [c_size_t_p, indvector_p,
										ct.c_size_t]
	lib.indvector_print.argtypes = [indvector_p]
	lib.indvector_indmin.argtypes = [indvector_p, c_size_t_p]
	lib.indvector_max.argtypes = [indvector_p, c_size_t_p]
	lib.indvector_min.argtypes = [indvector_p, c_size_t_p]

	lib.int_vector_alloc.argtypes = [int_vector_p, ct.c_size_t]
	lib.int_vector_calloc.argtypes = [int_vector_p, ct.c_size_t]
	lib.int_vector_free.argtypes = [int_vector_p]
	lib.int_vector_memcpy_vv.argtypes = [int_vector_p, int_vector_p]
	lib.int_vector_memcpy_va.argtypes = [int_vector_p, c_int_p, ct.c_size_t]
	lib.int_vector_memcpy_av.argtypes = [c_int_p, int_vector_p, ct.c_size_t]
	lib.int_vector_print.argtypes = [int_vector_p]

	## return values
	OptkitLibs.attach_default_restype(
			lib.vector_alloc,
			lib.vector_calloc,
			lib.vector_free,
			lib.vector_set_all,
			lib.vector_subvector,
			lib.vector_view_array,
			lib.vector_memcpy_vv,
			lib.vector_memcpy_va,
			lib.vector_memcpy_av,
			lib.vector_print,
			lib.vector_scale,
			lib.vector_add,
			lib.vector_sub,
			lib.vector_mul,
			lib.vector_div,
			lib.vector_add_constant,
			lib.vector_abs,
			lib.vector_recip,
			lib.vector_sqrt,
			lib.vector_pow,
			lib.vector_exp,
			lib.vector_indmin,
			lib.vector_min,
			lib.vector_max,
	)
	OptkitLibs.attach_default_restype(
			lib.indvector_alloc,
			lib.indvector_calloc,
			lib.indvector_free,
			lib.indvector_set_all,
			lib.indvector_subvector,
			lib.indvector_view_array,
			lib.indvector_memcpy_vv,
			lib.indvector_memcpy_va,
			lib.indvector_memcpy_av,
			lib.indvector_print,
			lib.indvector_indmin,
			lib.indvector_min,
			lib.indvector_max,
	)
	OptkitLibs.attach_default_restype(
			lib.int_vector_alloc,
			lib.int_vector_calloc,
			lib.int_vector_free,
			lib.int_vector_memcpy_vv,
			lib.int_vector_memcpy_va,
			lib.int_vector_memcpy_av,
			lib.int_vector_print,
	)

def attach_dense_linsys_ccalls(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	indvector_p = lib.indvector_p
	int_vector_p = lib.int_vector_p
	matrix_p = lib.matrix_p

	# Matrix
	# ------
	## arguments
	lib.matrix_alloc.argtypes = [
			matrix_p, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.matrix_calloc.argtypes = [
			matrix_p, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.matrix_free.argtypes = [matrix_p]
	lib.matrix_submatrix.argtypes = [
			matrix_p, matrix_p, ct.c_size_t, ct.c_size_t, ct.c_size_t,
			ct.c_size_t]
	lib.matrix_row.argtypes = [vector_p, matrix_p, ct.c_size_t]
	lib.matrix_column.argtypes = [vector_p, matrix_p, ct.c_size_t]
	lib.matrix_diagonal.argtypes = [vector_p, matrix_p]
	lib.matrix_view_array.argtypes = [
			matrix_p, ok_float_p, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.matrix_set_all.argtypes = [matrix_p, ok_float]
	lib.matrix_memcpy_mm.argtypes = [matrix_p, matrix_p]
	lib.matrix_memcpy_ma.argtypes = [matrix_p, ok_float_p, ct.c_uint]
	lib.matrix_memcpy_am.argtypes = [ok_float_p, matrix_p, ct.c_uint]
	lib.matrix_print.argtypes = [matrix_p]
	lib.matrix_scale.argtypes = [matrix_p, ok_float]
	lib.matrix_scale_left.argtypes = [matrix_p, vector_p]
	lib.matrix_scale_right.argtypes = [matrix_p, vector_p]
	lib.matrix_abs.argtypes = [matrix_p]
	lib.matrix_pow.argtypes = [matrix_p, ok_float]

	## return values
	OptkitLibs.attach_default_restype(
			lib.matrix_alloc,
			lib.matrix_calloc,
			lib.matrix_free,
			lib.matrix_submatrix,
			lib.matrix_row,
			lib.matrix_column,
			lib.matrix_diagonal,
			lib.matrix_view_array,
			lib.matrix_set_all,
			lib.matrix_memcpy_mm,
			lib.matrix_memcpy_ma,
			lib.matrix_memcpy_am,
			lib.matrix_print,
			lib.matrix_scale,
			lib.matrix_scale_left,
			lib.matrix_scale_right,
			lib.matrix_abs,
			lib.matrix_pow,
	)

	# BLAS
	# ----

	## arguments
	lib.blas_make_handle.argtypes = [ct.c_void_p]
	lib.blas_destroy_handle.argtypes = [ct.c_void_p]
	lib.blas_axpy.argtypes = [ct.c_void_p, ok_float, vector_p, vector_p]
	lib.blas_nrm2.argtypes = [ct.c_void_p, vector_p, ok_float_p]
	lib.blas_scal.argtypes = [ct.c_void_p, ok_float, vector_p]
	lib.blas_asum.argtypes = [ct.c_void_p, vector_p, ok_float_p]
	lib.blas_dot.argtypes = [ct.c_void_p, vector_p, vector_p, ok_float_p]
	lib.blas_gemv.argtypes = [
			ct.c_void_p, ct.c_uint, ok_float, matrix_p, vector_p, ok_float,
			vector_p]
	lib.blas_trsv.argtypes = [
			ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint, matrix_p, vector_p]
	lib.blas_sbmv.argtypes = [
			ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_size_t, ok_float, vector_p,
			vector_p, ok_float, vector_p]
	lib.blas_diagmv.argtypes = [
			ct.c_void_p, ok_float, vector_p, vector_p, ok_float, vector_p]
	lib.blas_syrk.argtypes = [
			ct.c_void_p, ct.c_uint, ct.c_uint, ok_float, matrix_p, ok_float,
			matrix_p]
	lib.blas_gemm.argtypes = [
			ct.c_void_p, ct.c_uint, ct.c_uint, ok_float, matrix_p, matrix_p,
			ok_float, matrix_p]
	lib.blas_trsm.argtypes = [
			ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ok_float,
			matrix_p, matrix_p]

	## return values
	OptkitLibs.attach_default_restype(
			lib.blas_make_handle,
			lib.blas_destroy_handle,
			lib.blas_axpy,
			lib.blas_nrm2,
			lib.blas_scal,
			lib.blas_asum,
			lib.blas_dot,
			lib.blas_gemv,
			lib.blas_trsv,
			lib.blas_sbmv,
			lib.blas_diagmv,
			lib.blas_syrk,
			lib.blas_gemm,
			lib.blas_trsm,
	)

	# LAPACK
	# ------
	lib.lapack_make_handle.argtypes = [ct.c_void_p]
	lib.lapack_destroy_handle.argtypes = [ct.c_void_p]
	lib.lapack_solve_LU.argtypes = [ct.c_void_p, matrix_p, vector_p, int_vector_p]

	OptkitLibs.attach_default_restype(
			lib.lapack_make_handle,
			lib.lapack_destroy_handle,
			lib.lapack_solve_LU,
	)

	# LINALG
	# -------
	## arguments
	lib.linalg_cholesky_decomp.argtypes = [ct.c_void_p, matrix_p]
	lib.linalg_cholesky_svx.argtypes = [ct.c_void_p, matrix_p, vector_p]
	lib.linalg_matrix_row_squares.argtypes = [ct.c_uint, matrix_p, vector_p]
	lib.linalg_matrix_broadcast_vector.argtypes = [
			matrix_p, vector_p, ct.c_uint, ct.c_uint]
	lib.linalg_matrix_reduce_indmin.argtypes = [
			indvector_p, vector_p, matrix_p, ct.c_uint]
	lib.linalg_matrix_reduce_min.argtypes = [vector_p, matrix_p, ct.c_uint]
	lib.linalg_matrix_reduce_max.argtypes = [vector_p, matrix_p, ct.c_uint]

	## return values
	OptkitLibs.attach_default_restype(
			lib.linalg_cholesky_decomp,
			lib.linalg_cholesky_svx,
			lib.linalg_matrix_row_squares,
			lib.linalg_matrix_broadcast_vector,
			lib.linalg_matrix_reduce_indmin,
			lib.linalg_matrix_reduce_min,
			lib.linalg_matrix_reduce_max,
	)

def attach_sparse_linsys_ccalls(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)
	include_ok_sparse(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	ok_int_p = lib.ok_int_p
	vector_p = lib.vector_p
	sparse_matrix_p = lib.sparse_matrix_p

	# Sparse Handle
	# -------------
	## arguments
	lib.sp_make_handle.argtypes = [ct.c_void_p]
	lib.sp_destroy_handle.argtypes = [ct.c_void_p]

	## return values
	lib.sp_make_handle.restype = ct.c_uint
	lib.sp_destroy_handle.restype = ct.c_uint

	# Sparse Matrix
	# -------------
	## arguments
	lib.sp_matrix_alloc.argtypes = [
			sparse_matrix_p, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.sp_matrix_calloc.argtypes = [
			sparse_matrix_p, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.sp_matrix_free.argtypes = [sparse_matrix_p]
	lib.sp_matrix_memcpy_mm.argtypes = [sparse_matrix_p, sparse_matrix_p]
	lib.sp_matrix_memcpy_ma.argtypes = [
			ct.c_void_p, sparse_matrix_p, ok_float_p, ok_int_p, ok_int_p]
	lib.sp_matrix_memcpy_am.argtypes = [
			ok_float_p, ok_int_p, ok_int_p, sparse_matrix_p]
	lib.sp_matrix_memcpy_vals_mm.argtypes = [sparse_matrix_p, sparse_matrix_p]
	lib.sp_matrix_memcpy_vals_ma.argtypes = [
			ct.c_void_p, sparse_matrix_p, ok_float_p]
	lib.sp_matrix_memcpy_vals_am.argtypes = [ok_float_p,sparse_matrix_p]
	lib.sp_matrix_abs.argtypes = [sparse_matrix_p]
	lib.sp_matrix_pow.argtypes = [sparse_matrix_p, ok_float]
	lib.sp_matrix_scale.argtypes = [sparse_matrix_p, ok_float]
	lib.sp_matrix_scale_left.argtypes = [
			ct.c_void_p, sparse_matrix_p, vector_p]
	lib.sp_matrix_scale_right.argtypes = [
			ct.c_void_p, sparse_matrix_p, vector_p]
	lib.sp_matrix_print.argtypes = [sparse_matrix_p]
	# lib.sp_matrix_print_transpose.argtypes = [sparse_matrix_p]

	## return values
	OptkitLibs.attach_default_restype(
			lib.sp_matrix_alloc,
			lib.sp_matrix_calloc,
			lib.sp_matrix_free,
			lib.sp_matrix_memcpy_mm,
			lib.sp_matrix_memcpy_ma,
			lib.sp_matrix_memcpy_am,
			lib.sp_matrix_memcpy_vals_mm,
			lib.sp_matrix_memcpy_vals_ma,
			lib.sp_matrix_memcpy_vals_am,
			lib.sp_matrix_abs,
			lib.sp_matrix_pow,
			lib.sp_matrix_scale,
			lib.sp_matrix_scale_left,
			lib.sp_matrix_scale_right,
			lib.sp_matrix_print,
	)
	# lib.sp_matrix_print_transpose.restype = None

	# Sparse BLAS
	# -----------
	## arguments
	lib.sp_blas_gemv.argtypes = [ct.c_void_p, ct.c_uint, ok_float, sparse_matrix_p,
								 vector_p, ok_float, vector_p]

	## return values
	OptkitLibs.attach_default_restype(lib.sp_blas_gemv)
