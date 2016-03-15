from ctypes import c_uint, c_size_t, c_void_p, CFUNCTYPE, POINTER, Structure
from optkit.libs.loader import retrieve_libs, validate_lib

class OperatorLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('liboperator_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, denselib, sparselib, single_precision=False, gpu=False):
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
			matrix_p = denselib.matrix_p
			sparse_matrix_p = sparselib.sparse_matrix_p

			class OperatorEnums(object):
				NULL = 0
				IDENTITY = 101
				DENSE = 201
				SPARSE_CSR = 301
				SPARSE_CSC = 302
				SPARSE_COO = 303

			lib.enums = OperatorEnums()

			class operator(Structure):
				_fields_ = [('size1', c_size_t),
							('size2', c_size_t),
							('data', c_void_p),
							('apply', CFUNCTYPE(None, c_void_p, vector_p,
												vector_p)),
							('adjoint', CFUNCTYPE(None, c_void_p,vector_p,
												  vector_p)),
							('fused_apply', CFUNCTYPE(None, c_void_p, ok_float,
													  vector_p, ok_float,
													  vector_p)),
							('fused_adjoint', CFUNCTYPE(None, c_void_p,
														ok_float, vector_p,
														ok_float, vector_p)),
							('free', CFUNCTYPE(None, c_void_p)),
							('kind', c_uint)]

			lib.operator = operator
			lib.operator_p = operator_p = POINTER(operator)
			lib.operator_pp = operator_pp = POINTER(operator_p)

			# argument types
			lib.operator_free.argtypes = [operator_p]
			lib.dense_operator_alloc.argtypes = [matrix_p]
			lib.sparse_operator_alloc.argtypes = [sparse_matrix_p]

			# return types
			lib.operator_free.restype = None
			lib.dense_operator_alloc.restype = operator_p
			lib.sparse_operator_alloc.restype = operator_p

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib