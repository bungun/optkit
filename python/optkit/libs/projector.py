from ctypes import Structure, CFUNCTYPE, POINTER, c_int, c_uint, c_size_t, \
				   c_void_p
from optkit.libs.loader import retrieve_libs, validate_lib

class ProjectorLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libprojector_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, denselib, operatorlib=None, single_precision=False,
			gpu=False):
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

			class ProjectorEnums(object):
				DENSE_DIRECT = 101
				SPARSE_DIRECT = 102
				INDIRECT = 103

			lib.enums = ProjectorEnums()

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

			class projector(Structure):
				_fields_ = [('kind', c_uint),
							('size1', c_size_t),
							('size2', c_size_t),
							('data', c_void_p),
							('initialize', CFUNCTYPE(None, c_void_p,
													 c_int)),
							('project', CFUNCTYPE(None, c_void_p, vector_p,
												  vector_p, vector_p, vector_p,
												  ok_float)),
							('free', CFUNCTYPE(None, c_void_p))]

			projector_p = POINTER(projector)
			lib.projector = projector
			lib.projector_p = projector_p

			lib.projector_free.argtypes = [projector_p]
			lib.projector_free.restypes = None

			class dense_direct_projector(Structure):
				_fields_ = [('A', matrix_p),
							('L', matrix_p),
							('linalg_handle', c_void_p),
							('normA', ok_float),
							('skinny', c_int),
							('normalized', c_int)]

			dense_direct_projector_p = POINTER(dense_direct_projector)
			lib.dense_direct_projector = dense_direct_projector
			lib.dense_direct_projector_p = dense_direct_projector_p

			lib.dense_direct_projector_alloc.argtypes = [matrix_p]
			lib.dense_direct_projector_alloc.restype = projector_p

			if operatorlib is not None:
				operator_p = operatorlib.operator_p

				class indirect_projector(Structure):
					_fields_ = [('A', operator_p),
								('cgls_work', c_void_p)]

				indirect_projector_p = POINTER(indirect_projector)
				lib.indirect_projector = indirect_projector
				lib.indirect_projector_p = indirect_projector_p


				lib.indirect_projector_alloc.argtypes = [
						indirect_projector_p, operator_p]
				lib.indirect_projector_initialize.argtypes = [
						c_void_p, indirect_projector_p, c_int]
				lib.indirect_projector_project.argtypes = [
						c_void_p, indirect_projector_p, vector_p, vector_p,
						vector_p, vector_p]
				lib.indirect_projector_free.argtypes = [indirect_projector_p]

				lib.indirect_projector_alloc.restype = None
				lib.indirect_projector_initialize.restype = None
				lib.indirect_projector_project.restype = None
				lib.indirect_projector_free.restype = None

				class indirect_projector_generic(Structure):
					_fields_ = [('A', operator_p),
								('cgls_work', c_void_p),
								('linalg_handle', c_void_p),
								('normA', ok_float),
								('normalized', c_int)]

				indirect_projector_generic_p = POINTER(
						indirect_projector_generic)
				lib.indirect_projector_generic = indirect_projector_generic
				lib.indirect_projector_generic_p = indirect_projector_generic_p

				lib.indirect_projector_generic_alloc.argtypes = [operator_p]
				lib.indirect_projector_generic_alloc.restype = projector_p

			else:
				lib.indirect_projector = AttributeError()
				lib.indirect_projector_p = AttributeError()
				lib.indirect_projector_alloc = AttributeError()
				lib.indirect_projector_initialize = AttributeError()
				lib.indirect_projector_project = AttributeError()
				lib.indirect_projector_free = AttributeError()

				lib.indirect_projector_generic = AttributeError()
				lib.indirect_projector_generic_p = AttributeError()
				lib.indirect_projector_generic_alloc = AttributeError()

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib