from ctypes import Structure, POINTER, c_int, c_uint, c_size_t, c_void_p
from optkit.libs.loader import retrieve_libs, validate_lib

class ProjectorLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libprojector_')
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
			ok_float = denselib.ok_float
			ok_float_p = denselib.ok_float_p
			vector_p = denselib.vector_p
			matrix_p = denselib.matrix_p

			class direct_projector(Structure):
				_fields_ = [('A', matrix_p),
							('L', matrix_p),
							('normA', ok_float),
							('skinny', c_int),
							('normalized', c_int)]

			direct_projector_p = POINTER(direct_projector)
			lib.direct_projector = direct_projector
			lib.direct_projector_p = direct_projector_p

			lib.direct_projector_alloc.argtypes = [direct_projector_p, matrix_p]
			lib.direct_projector_initialize.argtypes = [c_void_p,
				direct_projector_p,c_int]
			lib.direct_projector_project.argtypes = [c_void_p,
				direct_projector_p, vector_p, vector_p, vector_p, vector_p]
			lib.direct_projector_free.argtypes = [direct_projector_p]

			lib.direct_projector_alloc.restype = None
			lib.direct_projector_initialize.restype = None
			lib.direct_projector_project.restype = None
			lib.direct_projector_free.restype = None

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib