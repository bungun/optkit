from ctypes import CDLL, c_int, c_uint, c_size_t, c_void_p
from subprocess import check_output
from os import path, uname, getenv
from numpy import float32
from site import getsitepackages

class DenseLinsysLibs(object):
	def __init__(self):
		self.libs = {}
		local_c_build = path.abspath(path.join(path.dirname(__file__),
			'..', '..', '..', 'build'))
		search_results = ""
		use_local = getenv('OPTKIT_USE_LOCALLIBS', 0)

		# NB: no windows support
		ext = "dylib" if uname()[0] == "Darwin" else "so"
		for device in ['gpu', 'cpu']:
			for precision in ['32', '64']:
				for order in ['', 'col_', 'row_']:
					lib_tag = '{}{}{}'.format(order, device, precision)
					lib_name = 'libok_dense_{}{}{}.{}'.format(
						order, device, precision, ext)
					lib_path = getsitepackages()[0]
					if not use_local and path.exists(path.join(lib_path, lib_name)):
						lib_path = path.join(lib_path, lib_name)
					else:
						lib_path = path.join(local_c_build, lib_name)

					try:
						lib = CDLL(lib_path)
						self.libs[lib_tag]=lib
					except (OSError, IndexError):
						search_results += str("library {} not found at {}.\n".format(
							lib_name, lib_path))
						self.libs[lib_tag] = None

		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, lowtypes, GPU=False):
		device = 'gpu' if GPU else 'cpu'
		precision = '32' if lowtypes.FLOAT_CAST == float32 else '64'
		lib_key = '{}'.format(lowtypes.order)
		if lib_key != '': lib_key += '_'
		lib_key += '{}{}'.format(device, precision)

		if self.libs[lib_key] is not None:
			lib = self.libs[lib_key]
			ok_float = lowtypes.ok_float
			ok_float_p = lowtypes.ok_float_p
			vector_p = lowtypes.vector_p
			matrix_p = lowtypes.matrix_p
			# Vector 
			# ------
			## arguments
			lib.vector_alloc.argtypes = [vector_p, c_size_t]
			lib.vector_calloc.argtypes = [vector_p, c_size_t]
			lib.vector_free.argtypes = [vector_p]
			lib.vector_set_all.argtypes = [vector_p, ok_float]
			lib.vector_subvector.argtypes = [vector_p, vector_p, c_size_t, c_size_t]
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

			# Matrix
			# ------
			## arguments
			lib.matrix_alloc.argtypes = [matrix_p, c_size_t, c_size_t, c_uint]
			lib.matrix_calloc.argtypes = [matrix_p, c_size_t, c_size_t, c_uint]
			lib.matrix_free.argtypes = [matrix_p]
			lib.matrix_submatrix.argtypes = [matrix_p, matrix_p, c_size_t, c_size_t, c_size_t, c_size_t]
			lib.matrix_row.argtypes = [vector_p, matrix_p, c_size_t]
			lib.matrix_column.argtypes = [vector_p, matrix_p, c_size_t]
			lib.matrix_diagonal.argtypes = [vector_p, matrix_p]
			lib.matrix_view_array.argtypes = [matrix_p, ok_float_p, c_size_t, c_size_t, c_uint]
			lib.matrix_set_all.argtypes = [matrix_p, ok_float]
			lib.matrix_memcpy_mm.argtypes = [matrix_p, matrix_p]
			lib.matrix_memcpy_ma.argtypes = [matrix_p, ok_float_p, c_uint]
			lib.matrix_memcpy_am.argtypes = [ok_float_p, matrix_p, c_uint]
			lib.matrix_print.argtypes = [matrix_p]
			lib.matrix_scale.argtypes = [matrix_p, ok_float]

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
			lib.blas_gemv.argtypes = [c_void_p, c_uint, ok_float, matrix_p, vector_p, ok_float, vector_p]
			lib.blas_trsv.argtypes = [c_void_p, c_uint, c_uint, c_uint, matrix_p, vector_p]
			lib.blas_syrk.argtypes = [c_void_p, c_uint, c_uint, ok_float, matrix_p, ok_float, matrix_p]
			lib.blas_gemm.argtypes = [c_void_p, c_uint, c_uint, ok_float, matrix_p, matrix_p, ok_float, matrix_p]
			lib.blas_trsm.argtypes = [c_void_p, c_uint, c_uint, c_uint, c_uint, ok_float, matrix_p, matrix_p]

			## return values 
			lib.blas_make_handle.restype = None
			lib.blas_destroy_handle.restype = None
			lib.blas_axpy.restype = None
			lib.blas_nrm2.restype = ok_float
			lib.blas_scal.restype = None
			lib.blas_asum.restype = ok_float
			lib.blas_dot.restype = ok_float
			lib.blas_gemv.restype = None
			lib.blas_trsv.restype = None
			lib.blas_syrk.restype = None
			lib.blas_gemm.restype = None
			lib.blas_trsm.restype = None

			# LINALG
			# -------
			## arguments
			lib.linalg_cholesky_decomp.argtypes = [c_void_p, matrix_p]
			lib.linalg_cholesky_svx.argtypes = [c_void_p, matrix_p, vector_p]

			## return values
			lib.linalg_cholesky_decomp.restype = None
			lib.linalg_cholesky_svx.restype = None

			# DEVICE
			# ------
			lib.ok_device_reset.argtypes = []
			lib.ok_device_reset.restype = None

			return lib


class SparseLinsysLibs(object):
	def __init__(self):
		self.libs = {}
		local_c_build = path.abspath(path.join(path.dirname(__file__),
			'..', '..', '..', 'build'))
		search_results = ""
		use_local = getenv('OPTKIT_USE_LOCALLIBS', 0)

		# NB: no windows support
		ext = "dylib" if uname()[0] == "Darwin" else "so"
		for device in ['gpu', 'cpu']:
			for precision in ['32', '64']:
				for order in ['', 'col_', 'row_']:
					lib_tag = '{}{}{}'.format(order, device, precision)
					lib_name = 'libok_sparse_{}{}{}.{}'.format(
						order, device, precision, ext)
					lib_path = getsitepackages()[0]
					if not use_local and path.exists(path.join(lib_path, lib_name)):
						lib_path = path.join(lib_path, lib_name)
					else:
						lib_path = path.join(local_c_build, lib_name)

					try:
						lib = CDLL(lib_path)
						self.libs[lib_tag]=lib
					except (OSError, IndexError):
						search_results += str("library {} not found at {}.\n".format(
							lib_name, lib_path))
						self.libs[lib_tag] = None

		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, lowtypes, GPU=False):
		device = 'gpu' if GPU else 'cpu'
		precision = '32' if lowtypes.FLOAT_CAST == float32 else '64'
		lib_key = '{}'.format(lowtypes.order)
		if lib_key != '': lib_key += '_'
		lib_key += '{}{}'.format(device, precision)

		if self.libs[lib_key] is not None:
			lib = self.libs[lib_key]
			ok_float = lowtypes.ok_float
			ok_float_p = lowtypes.ok_float_p
			ok_int = lowtypes.ok_int
			ok_int_p = lowtypes.ok_int_p
			vector_p = lowtypes.vector_p
			sparse_matrix_p = lowtypes.sparse_matrix_p

			# Sparse Handle
			# -------------
			## arguments
			lib.sp_make_handle.argtypes = [c_void_p]
			lib.sp_destroy_handle.argtypes = [c_void_p]

			## return values 
			lib.sp_make_handle.restype = None
			lib.sp_destroy_handle.restype = None


			# Sparse Matrix 
			# -------------

			## arguments
			lib.sp_matrix_alloc.argtypes = [sparse_matrix_p, c_size_t, c_size_t, c_size_t, c_uint]
			lib.sp_matrix_calloc.argtypes = [sparse_matrix_p, c_size_t, c_size_t, c_size_t, c_uint]
			lib.sp_matrix_free.argtypes = [sparse_matrix_p]
			lib.sp_matrix_memcpy_mm.argtypes = [sparse_matrix_p, sparse_matrix_p]
			lib.sp_matrix_memcpy_ma.argtypes = [c_void_p, sparse_matrix_p, ok_float_p, ok_int_p, ok_int_p]
			lib.sp_matrix_memcpy_am.argtypes = [ok_float_p, ok_int_p, ok_int_p, sparse_matrix_p]
			lib.sp_matrix_memcpy_vals_mm.argtypes = [sparse_matrix_p, sparse_matrix_p]
			lib.sp_matrix_memcpy_vals_ma.argtypes = [c_void_p, sparse_matrix_p, ok_float_p]
			lib.sp_matrix_memcpy_vals_am.argtypes = [ok_float_p, sparse_matrix_p]
			lib.sp_matrix_memcpy_pattern_mm.argtypes = [sparse_matrix_p, sparse_matrix_p]
			lib.sp_matrix_memcpy_pattern_ma.argtypes = [c_void_p, sparse_matrix_p, ok_int_p, ok_int_p]
			lib.sp_matrix_memcpy_pattern_am.argtypes = [ok_int_p, ok_int_p, sparse_matrix_p]
			lib.sp_matrix_abs.argtypes = [sparse_matrix_p]
			lib.sp_matrix_scale.argtypes = [sparse_matrix_p, ok_float]
			lib.sp_matrix_scale_left.argtypes = [c_void_p, sparse_matrix_p, vector_p]
			lib.sp_matrix_scale_right.argtypes = [c_void_p, sparse_matrix_p, vector_p]
			lib.sp_matrix_print.argtypes = [sparse_matrix_p]

			## return values 
			lib.sp_make_handle.restype = None
			lib.sp_matrix_alloc.restype = None
			lib.sp_matrix_calloc.restype = None
			lib.sp_matrix_free.restype = None
			lib.sp_matrix_memcpy_mm.restype = None
			lib.sp_matrix_memcpy_ma.restype = None
			lib.sp_matrix_memcpy_am.restype = None
			lib.sp_matrix_memcpy_vals_mm.restype = None
			lib.sp_matrix_memcpy_vals_ma.restype = None
			lib.sp_matrix_memcpy_vals_am.restype = None
			lib.sp_matrix_memcpy_pattern_mm.restype = None
			lib.sp_matrix_memcpy_pattern_ma.restype = None
			lib.sp_matrix_memcpy_pattern_am.restype = None
			lib.sp_matrix_abs.restype = None
			lib.sp_matrix_scale.restype = None
			lib.sp_matrix_scale_left.restype = None
			lib.sp_matrix_scale_right.restype = None			
			lib.sp_matrix_print.restype = None

			# Sparse BLAS
			# -----------

			## arguments
			lib.sp_blas_gemv.argtypes = [c_void_p, c_uint, ok_float, sparse_matrix_p, vector_p, ok_float, vector_p]

			## return values 
			lib.sp_blas_gemv.restype = None

			return lib
