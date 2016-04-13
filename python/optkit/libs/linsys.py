from numpy import float32, float64
from ctypes import c_int, c_uint, c_size_t, c_void_p, c_float, c_double, \
	POINTER, Structure
from optkit.libs.loader import retrieve_libs, validate_lib

class DenseLinsysLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libok_dense_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, single_precision=False, gpu=False):
		device = 'gpu' if gpu else 'cpu'
		precision = '32' if single_precision else '64'
		lib_key = '{}{}'.format(device, precision)
		if lib_key not in self.libs:
			return None
		elif self.libs[lib_key] is None:
			return None

		lib = self.libs[lib_key]
		if lib.INITIALIZED:
			return lib
		else:
			class OKEnums(object):
				CblasRowMajor = c_uint(101).value
				CblasColMajor = c_uint(102).value
				CblasNoTrans = c_uint(111).value
				CblasTrans = c_uint(112).value
				CblasConjTrans = c_uint(113).value
				CblasUpper = c_uint(121).value
				CblasLower = c_uint(122).value
				CblasNonUnit = c_uint(131).value
				CblasUnit = c_uint(132).value
				CblasLeft = c_uint(141).value
				CblasRight = c_uint(142).value
				OkTransformScale = c_uint(0).value
				OkTransformAdd = c_uint(1).value

			lib.enums = OKEnums()

			# low-level c types, as defined for optkit
			# ----------------------------------------
			ok_float = lib.ok_float = c_float if single_precision else c_double
			ok_int = lib.ok_int = c_int

			lib.pyfloat = float32 if single_precision else float64

			# pointers to C types
			c_int_p = lib.c_int_p = POINTER(c_int)
			c_size_t_p = lib.c_size_t_p = POINTER(c_size_t)
			ok_float_p = lib.ok_float_p = POINTER(ok_float)
			ok_int_p = lib.ok_int_p = POINTER(ok_int)

			# low-level optkit types
			# ----------------------

			# vector struct
			class ok_vector(Structure):
				_fields_ = [('size', c_size_t),
							('stride', c_size_t),
							('data', ok_float_p)]
			lib.vector = ok_vector
			vector_p = lib.vector_p = POINTER(lib.vector)

			# index vector struct
			class ok_indvector(Structure):
				_fields_ = [('size', c_size_t),
							('stride', c_size_t),
							('data', c_size_t_p)]
			lib.indvector = ok_indvector
			indvector_p = lib.indvector_p = POINTER(lib.indvector)

			# matrix struct
			class ok_matrix(Structure):
				_fields_ = [('size1', c_size_t),
							('size2', c_size_t),
							('ld', c_size_t),
							('data', ok_float_p),
							('order', c_uint)]
			matrix = lib.matrix = ok_matrix
			matrix_p = lib.matrix_p = POINTER(lib.matrix)


			# Vector
			# ------
			## arguments
			lib.vector_alloc.argtypes = [vector_p, c_size_t]
			lib.vector_calloc.argtypes = [vector_p, c_size_t]
			lib.vector_free.argtypes = [vector_p]
			lib.vector_set_all.argtypes = [vector_p, ok_float]
			lib.vector_subvector.argtypes = [vector_p, vector_p, c_size_t,
												c_size_t]
			lib.vector_view_array.argtypes = [vector_p, ok_float_p, c_size_t]
			lib.vector_memcpy_vv.argtypes = [vector_p, vector_p]
			lib.vector_memcpy_va.argtypes = [vector_p, ok_float_p, c_size_t]
			lib.vector_memcpy_av.argtypes = [ok_float_p, vector_p, c_size_t]
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
			lib.vector_indmin.argtypes = [vector_p]
			lib.vector_min.argtypes = [vector_p]
			lib.vector_max.argtypes = [vector_p]

			lib.indvector_alloc.argtypes = [indvector_p, c_size_t]
			lib.indvector_calloc.argtypes = [indvector_p, c_size_t]
			lib.indvector_free.argtypes = [indvector_p]
			lib.indvector_set_all.argtypes = [indvector_p, ok_float]
			lib.indvector_subvector.argtypes = [indvector_p, indvector_p,
												c_size_t, c_size_t]
			lib.indvector_view_array.argtypes = [indvector_p, c_size_t_p,
												 c_size_t]
			lib.indvector_memcpy_vv.argtypes = [indvector_p, indvector_p]
			lib.indvector_memcpy_va.argtypes = [indvector_p, c_size_t_p,
												c_size_t]
			lib.indvector_memcpy_av.argtypes = [c_size_t_p, indvector_p,
												c_size_t]

			## return values
			lib.vector_alloc.restype = None
			lib.vector_calloc.restype = None
			lib.vector_free.restype = None
			lib.vector_set_all.restype = None
			lib.vector_subvector.restype = None
			lib.vector_view_array.restype = None
			lib.vector_memcpy_vv.restype = None
			lib.vector_memcpy_va.restype = None
			lib.vector_memcpy_av.restype = None
			lib.vector_print.restype = None
			lib.vector_scale.restype = None
			lib.vector_add.restype = None
			lib.vector_sub.restype = None
			lib.vector_mul.restype = None
			lib.vector_div.restype = None
			lib.vector_add_constant.restype = None
			lib.vector_abs.restype = None
			lib.vector_recip.restype = None
			lib.vector_sqrt.restype = None
			lib.vector_pow.restype = None
			lib.vector_exp.restype = None
			lib.vector_indmin.restype = c_size_t
			lib.vector_min.restype = ok_float
			lib.vector_max.restype = ok_float

			lib.indvector_alloc.restype = None
			lib.indvector_calloc.restype = None
			lib.indvector_free.restype = None
			lib.indvector_set_all.restype = None
			lib.indvector_subvector.restype = None
			lib.indvector_view_array.restype = None
			lib.indvector_memcpy_vv.restype = None
			lib.indvector_memcpy_va.restype = None
			lib.indvector_memcpy_av.restype = None

			# Matrix
			# ------
			## arguments
			lib.matrix_alloc.argtypes = [matrix_p, c_size_t, c_size_t, c_uint]
			lib.matrix_calloc.argtypes = [matrix_p, c_size_t, c_size_t, c_uint]
			lib.matrix_free.argtypes = [matrix_p]
			lib.matrix_submatrix.argtypes = [matrix_p, matrix_p, c_size_t,
												c_size_t, c_size_t, c_size_t]
			lib.matrix_row.argtypes = [vector_p, matrix_p, c_size_t]
			lib.matrix_column.argtypes = [vector_p, matrix_p, c_size_t]
			lib.matrix_diagonal.argtypes = [vector_p, matrix_p]
			lib.matrix_view_array.argtypes = [matrix_p, ok_float_p, c_size_t,
												c_size_t, c_uint]
			lib.matrix_set_all.argtypes = [matrix_p, ok_float]
			lib.matrix_memcpy_mm.argtypes = [matrix_p, matrix_p]
			lib.matrix_memcpy_ma.argtypes = [matrix_p, ok_float_p, c_uint]
			lib.matrix_memcpy_am.argtypes = [ok_float_p, matrix_p, c_uint]
			lib.matrix_print.argtypes = [matrix_p]
			lib.matrix_scale.argtypes = [matrix_p, ok_float]
			lib.matrix_scale_left.argtypes = [matrix_p, vector_p]
			lib.matrix_scale_right.argtypes = [matrix_p, vector_p]
			lib.matrix_abs.argtypes = [matrix_p]
			lib.matrix_pow.argtypes = [matrix_p, ok_float]

			## return values
			lib.matrix_alloc.restype = None
			lib.matrix_calloc.restype = None
			lib.matrix_free.restype = None
			lib.matrix_submatrix.restype = None
			lib.matrix_row.restype = None
			lib.matrix_column.restype = None
			lib.matrix_diagonal.restype = None
			lib.matrix_view_array.restype = None
			lib.matrix_set_all.restype = None
			lib.matrix_memcpy_mm.restype = None
			lib.matrix_memcpy_ma.restype = None
			lib.matrix_memcpy_am.restype = None
			lib.matrix_print.restype = None
			lib.matrix_scale.restype = None
			lib.matrix_scale_left.restype = None
			lib.matrix_scale_right.restype = None
			lib.matrix_abs.restype = None
			lib.matrix_pow.restype = None

			# BLAS
			# ----

			## arguments
			lib.blas_make_handle.argtypes = [c_void_p]
			lib.blas_destroy_handle.argtypes = [c_void_p]
			lib.blas_axpy.argtypes = [c_void_p, ok_float, vector_p, vector_p]
			lib.blas_nrm2.argtypes = [c_void_p, vector_p]
			lib.blas_scal.argtypes = [c_void_p, ok_float, vector_p]
			lib.blas_asum.argtypes = [c_void_p, vector_p]
			lib.blas_dot.argtypes = [c_void_p, vector_p, vector_p]
			lib.blas_gemv.argtypes = [c_void_p, c_uint, ok_float, matrix_p,
										vector_p, ok_float, vector_p]
			lib.blas_trsv.argtypes = [c_void_p, c_uint, c_uint, c_uint,
										matrix_p, vector_p]
			lib.blas_sbmv.argtypes = [c_void_p, c_uint, c_uint, c_size_t,
										ok_float, vector_p, vector_p, ok_float,
										vector_p]
			lib.blas_diagmv.argtypes = [c_void_p, ok_float, vector_p, vector_p,
										ok_float, vector_p]
			lib.blas_syrk.argtypes = [c_void_p, c_uint, c_uint, ok_float,
										matrix_p, ok_float, matrix_p]
			lib.blas_gemm.argtypes = [c_void_p, c_uint, c_uint, ok_float,
										matrix_p, matrix_p, ok_float, matrix_p]
			lib.blas_trsm.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint,
										ok_float, matrix_p, matrix_p]

			## return values
			lib.blas_make_handle.restype = c_uint
			lib.blas_destroy_handle.restype = c_uint
			lib.blas_axpy.restype = None
			lib.blas_nrm2.restype = ok_float
			lib.blas_scal.restype = None
			lib.blas_asum.restype = ok_float
			lib.blas_dot.restype = ok_float
			lib.blas_gemv.restype = None
			lib.blas_trsv.restype = None
			lib.blas_sbmv.restype = None
			lib.blas_diagmv.restype = None
			lib.blas_syrk.restype = None
			lib.blas_gemm.restype = None
			lib.blas_trsm.restype = None

			# LINALG
			# -------
			## arguments
			lib.linalg_cholesky_decomp.argtypes = [c_void_p, matrix_p]
			lib.linalg_cholesky_svx.argtypes = [c_void_p, matrix_p, vector_p]
			lib.linalg_diag_gramian.argtypes = [c_void_p, matrix_p, vector_p]
			lib.linalg_matrix_broadcast_vector.argtypes = [c_void_p, matrix_p,
													vector_p, c_uint, c_uint]
			lib.linalg_matrix_reduce_indmin.argtypes = [c_void_p, indvector_p,
														vector_p, matrix_p,
														c_uint]
			lib.linalg_matrix_reduce_min.argtypes = [c_void_p, vector_p,
													 matrix_p, c_uint]
			lib.linalg_matrix_reduce_max.argtypes = [c_void_p, vector_p,
													 matrix_p, c_uint]

			## return values
			lib.linalg_cholesky_decomp.restype = None
			lib.linalg_cholesky_svx.restype = None
			lib.linalg_diag_gramian.restype = None
			lib.linalg_matrix_broadcast_vector.restype = None
			lib.linalg_matrix_reduce_indmin.restype = None
			lib.linalg_matrix_reduce_min.restype = None
			lib.linalg_matrix_reduce_max.restype = None

			# DEVICE
			# ------
			lib.ok_device_reset.argtypes = []
			lib.ok_device_reset.restype = c_uint

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib


class SparseLinsysLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libok_sparse_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, denselib, single_precision=False, gpu=False):
		device = 'gpu' if gpu else 'cpu'
		precision = '32' if single_precision else '64'
		lib_key = '{}{}'.format(device, precision)

		if lib_key not in self.libs:
			return None
		elif self.libs[lib_key] is None:
			return None

		validate_lib(denselib, 'denselib', 'vector_calloc', type(self),
			single_precision, gpu)

		lib = self.libs[lib_key]
		if lib.INITIALIZED:
			return lib
		else:
			lib.pyfloat = denselib.pyfloat
			ok_float = denselib.ok_float
			ok_float_p = lib.ok_float_p = denselib.ok_float_p
			ok_int = denselib.ok_int
			ok_int_p = lib.ok_int_p = denselib.ok_int_p
			vector_p = denselib.vector_p

			# sparse matrix struct
			class ok_sparse_matrix(Structure):
				_fields_ = [('size1', c_size_t),
							('size2', c_size_t),
							('nnz', c_size_t),
							('ptrlen', c_size_t),
							('val', ok_float_p),
							('ind', ok_int_p),
							('ptr', ok_int_p),
							('order', c_uint)]

			lib.sparse_matrix = ok_sparse_matrix
			sparse_matrix_p = lib.sparse_matrix_p = POINTER(lib.sparse_matrix)

			# Sparse Handle
			# -------------
			## arguments
			lib.sp_make_handle.argtypes = [c_void_p]
			lib.sp_destroy_handle.argtypes = [c_void_p]

			## return values
			lib.sp_make_handle.restype = c_uint
			lib.sp_destroy_handle.restype = c_uint


			# Sparse Matrix
			# -------------
			## arguments
			lib.sp_matrix_alloc.argtypes = [sparse_matrix_p, c_size_t, c_size_t,
											c_size_t, c_uint]
			lib.sp_matrix_calloc.argtypes = [sparse_matrix_p, c_size_t,
											 c_size_t, c_size_t, c_uint]
			lib.sp_matrix_free.argtypes = [sparse_matrix_p]
			lib.sp_matrix_memcpy_mm.argtypes = [sparse_matrix_p,
												sparse_matrix_p]
			lib.sp_matrix_memcpy_ma.argtypes = [c_void_p, sparse_matrix_p,
												ok_float_p, ok_int_p, ok_int_p]
			lib.sp_matrix_memcpy_am.argtypes = [ok_float_p, ok_int_p, ok_int_p,
												sparse_matrix_p]
			lib.sp_matrix_memcpy_vals_mm.argtypes = [sparse_matrix_p,
													 sparse_matrix_p]
			lib.sp_matrix_memcpy_vals_ma.argtypes = [c_void_p, sparse_matrix_p,
													 ok_float_p]
			lib.sp_matrix_memcpy_vals_am.argtypes = [ok_float_p,
													 sparse_matrix_p]
			lib.sp_matrix_abs.argtypes = [sparse_matrix_p]
			lib.sp_matrix_pow.argtypes = [sparse_matrix_p, ok_float]
			lib.sp_matrix_scale.argtypes = [sparse_matrix_p, ok_float]
			lib.sp_matrix_scale_left.argtypes = [c_void_p, sparse_matrix_p,
												 vector_p]
			lib.sp_matrix_scale_right.argtypes = [c_void_p, sparse_matrix_p,
												  vector_p]
			lib.sp_matrix_print.argtypes = [sparse_matrix_p]
			# lib.sp_matrix_print_transpose.argtypes = [sparse_matrix_p]

			## return values
			lib.sp_matrix_alloc.restype = None
			lib.sp_matrix_calloc.restype = None
			lib.sp_matrix_free.restype = None
			lib.sp_matrix_memcpy_mm.restype = None
			lib.sp_matrix_memcpy_ma.restype = None
			lib.sp_matrix_memcpy_am.restype = None
			lib.sp_matrix_memcpy_vals_mm.restype = None
			lib.sp_matrix_memcpy_vals_ma.restype = None
			lib.sp_matrix_memcpy_vals_am.restype = None
			lib.sp_matrix_abs.restype = None
			lib.sp_matrix_pow.restype = None
			lib.sp_matrix_scale.restype = None
			lib.sp_matrix_scale_left.restype = None
			lib.sp_matrix_scale_right.restype = None
			lib.sp_matrix_print.restype = None
			# lib.sp_matrix_print_transpose.restype = None

			# Sparse BLAS
			# -----------

			## arguments
			lib.sp_blas_gemv.argtypes = [c_void_p, c_uint, ok_float,
										 sparse_matrix_p, vector_p, ok_float,
										 vector_p]

			## return values
			lib.sp_blas_gemv.restype = None

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True

			return lib