from ctypes import c_int, c_uint, c_size_t, c_void_p, CFUNCTYPE, POINTER, \
				   Structure
from optkit.libs.loader import retrieve_libs, validate_lib

class ConjugateGradientLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libcg_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, denselib, operatorlib, single_precision=False, gpu=False):
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
			ok_float = denselib.ok_float
			vector_p = denselib.vector_p
			operator_p = operatorlib.operator_p

			class cgls_helper(Structure):
				_fields_ = [('p', vector_p),
							('q', vector_p),
							('r', vector_p),
							('s', vector_p),
							('norm_s', ok_float),
							('norm_s0', ok_float),
							('norm_x', ok_float),
							('xmax', ok_float),
							('alpha', ok_float),
							('beta', ok_float),
							('delta', ok_float),
							('gamma', ok_float),
							('gamma_prev', ok_float),
							('shrink', ok_float),
							('blas_handle', c_void_p)]

			lib.cgls_helper = cgls_helper
			lib.cgls_helper_p = cgls_helper_p = POINTER(cgls_helper)

			class pcg_helper(Structure):
				_fields_ = [('p', vector_p),
							('q', vector_p),
							('r', vector_p),
							('z', vector_p),
							('temp', vector_p),
							('norm_r', ok_float),
							('alpha', ok_float),
							('gamma', ok_float),
							('gamma_prev', ok_float),
							('blas_handle', c_void_p),
							('never_solved', c_int)]

			lib.pcg_helper = pcg_helper
			lib.pcg_helper_p = pcg_helper_p = POINTER(pcg_helper)

			# argument types
			lib.cgls_helper_alloc.argtypes = [c_size_t, c_size_t]
			lib.cgls_helper_free.argtypes = [cgls_helper_p]

			lib.cgls_nonallocating.argtypes = [cgls_helper_p, operator_p,
											   operator_p, vector_p, vector_p,
											   ok_float, ok_float, c_size_t,
											   c_int]
			lib.cgls.argtypes = [operator_p, vector_p, vector_p, ok_float,
								 ok_float, c_size_t, c_int]
			lib.cgls_easy_init.argtypes = [c_size_t, c_size_t]
			lib.cgls_easy_solve.argtypes = [c_void_p, operator_p, operator_p,
											ok_float, ok_float, c_size_t,
											c_int]
			lib.cgls_easy_finish.argtypes = [c_void_p]

			lib.pcg_helper_alloc.argtypes = [c_size_t, c_size_t]
			lib.pcg_helper_free.argtypes = [pcg_helper_p]

			lib.pcg_nonallocating.argtypes = [pcg_helper_p, operator_p,
											  operator_p, vector_p, vector_p,
											  ok_float, ok_float, c_size_t,
											  c_int]
			lib.pcg.argtypes = [operator_p, operator_p, vector_p, vector_p,
								ok_float, ok_float, c_size_t, c_int]
			lib.pcg_easy_init.argtypes = [c_size_t, c_size_t]
			lib.pcg_easy_solve.argtypes = [c_void_p, operator_p, operator_p,
										   vector_p, vector_p, ok_float,
										   ok_float, c_size_t, c_int]
			lib.pcg_easy_finish.argtypes = [c_void_p]

			# return types
			lib.cgls_helper_alloc.restype = cgls_helper_p
			lib.cgls_helper_free.retype = None

			lib.cgls_nonallocating.restype = c_uint
			lib.cgls.restype = c_uint
			lib.cgls_easy_init.restype = c_void_p
			lib.cgls_easy_solve.restype = c_uint
			lib.cgls_easy_finish.restype = None

			lib.pcg_helper_alloc.restype = pcg_helper_p
			lib.pcg_helper_free.retype = None

			lib.pcg_nonallocating.restype = c_uint
			lib.pcg.restype = c_uint
			lib.pcg_easy_init.restype = c_void_p
			lib.pcg_easy_solve.restype = c_uint
			lib.pcg_easy_finish.restype = None


			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib