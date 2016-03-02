from ctypes import c_uint, c_size_t, c_void_p
from optkit.libs.loader import retrieve_libs, validate_lib

class EquilibrationLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libequil_')
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
			ok_float_p = denselib.ok_float_p
			vector_p = denselib.vector_p
			matrix_p = denselib.matrix_p

			# argument types
			lib.sinkhorn_knopp.argtypes = [c_void_p, ok_float_p, matrix_p,
												vector_p, vector_p, c_uint]
			lib.dense_l2.argtypes = [c_void_p, ok_float_p, matrix_p,
												vector_p, vector_p, c_uint]

			# return types
			lib.sinkhorn_knopp.restype = None
			lib.dense_l2.restype = None

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib