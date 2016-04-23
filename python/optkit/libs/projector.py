from ctypes import Structure, CFUNCTYPE, POINTER, c_int, c_uint, c_size_t, \
				   c_void_p
from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_sparse_linsys_ctypes, attach_base_ccalls, attach_vector_ccalls, \
	attach_dense_linsys_ccalls, attach_sparse_linsys_ccalls
from optkit.libs.operator import attach_operator_ctypes, attach_operator_ccalls
from optkit.libs.cg import attach_cg_ctypes, attach_cg_ccalls

class ProjectorLibs(object):
	def __init__(self):
		OptkitLibs.__init__(self, 'libprojector_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_sparse_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_dense_linsys_ccalls)
		self.attach_calls.append(attach_sparse_linsys_ccalls)
		self.attach_calls.append(attach_operator_ctypes)
		self.attach_calls.append(attach_operator_ccalls)
		self.attach_calls.append(attach_cg_ctypes)
		self.attach_calls.append(attach_cg_ccalls)
		self.attach_calls.append(attach_projector_ctypes)
		self.attach_calls.append(attach_projector_ccalls)
		self.attach_calls.append(attach_operator_projector_ctypes_ccalls)

def attach_projector_ctypes(lib, single_precision=False):
	if 'matrix_p' not in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p

	class direct_projector(Structure):
		_fields_ = [('A', matrix_p),
					('L', matrix_p),
					('normA', ok_float),
					('skinny', c_int),
					('normalized', c_int)]

	lib.direct_projector = direct_projector
	lib.direct_projector_p = POINTER(lib.direct_projector)

	class indirect_projector(Structure):
		_fields_ = [('A', operator_p),
					('cgls_work', c_void_p)]

	lib.indirect_projector = indirect_projector
	lib.indirect_projector_p = POINTER(lib.indirect_projector)

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

	lib.projector = projector
	lib.projector_p = POINTER(lib.projector)

	class dense_direct_projector(Structure):
		_fields_ = [('A', matrix_p),
					('L', matrix_p),
					('linalg_handle', c_void_p),
					('normA', ok_float),
					('skinny', c_int),
					('normalized', c_int)]

	lib.dense_direct_projector = dense_direct_projector
	lib.dense_direct_projector_p = POINTER(lib.dense_direct_projector)

def attach_projector_ccalls(lib, single_precision=False):
	if 'matrix_p' not in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if 'projector_p' not in lib.__dict__:
		attach_projector_ctypes(lib, single_precision)

	vector_p = lib.vector_p
	matrix_p = lib.matrix_p
	direct_projector_p = lib.direct_projector_p
	indirect_projector_p = lib.indirect_projector_p
	projector_p = lib.projector_p

	# args:
	# -direct
	lib.direct_projector_alloc.argtypes = [direct_projector_p, matrix_p]
	lib.direct_projector_initialize.argtypes = [c_void_p, direct_projector_p,
												c_int]
	lib.direct_projector_project.argtypes = [c_void_p,
		direct_projector_p, vector_p, vector_p, vector_p, vector_p]
	lib.direct_projector_free.argtypes = [direct_projector_p]
	lib.dense_direct_projector_alloc.argtypes = [matrix_p]

	# -indirect
	lib.indirect_projector_alloc.argtypes = [indirect_projector_p, operator_p]
	lib.indirect_projector_initialize.argtypes = [c_void_p,
												  indirect_projector_p, c_int]
	lib.indirect_projector_project.argtypes = [c_void_p, indirect_projector_p,
											   vector_p, vector_p, vector_p,
											   vector_p]
	lib.indirect_projector_free.argtypes = [indirect_projector_p]

	# -generic
	lib.projector_free.argtypes = [projector_p]

	# returns:
	# -direct
	lib.direct_projector_alloc.restype = None
	lib.direct_projector_initialize.restype = None
	lib.direct_projector_project.restype = None
	lib.direct_projector_free.restype = None
	lib.dense_direct_projector_alloc.restype = projector_p

	# -indirect
	lib.indirect_projector_alloc.restype = None
	lib.indirect_projector_initialize.restype = None
	lib.indirect_projector_project.restype = None
	lib.indirect_projector_free.restype = None

	# -generic
	lib.projector_free.restypes = None

def attach_operator_projector_ctypes_ccalls(lib, single_precision=False):
	if 'ok_float' not in lib.__dict__:
		attach_base_ctypes(lib, single_precision)
	if 'operator_p' not in lib.__dict__:
		attach_operator_ctypes(lib, single_precision)
	if 'projector_p' not in lib.__dict__:
		attach_projector_ctypes(lib, single_precision)


	class indirect_projector_generic(Structure):
		_fields_ = [('A', operator_p),
					('cgls_work', c_void_p),
					('linalg_handle', c_void_p),
					('normA', ok_float),
					('normalized', c_int)]

	lib.indirect_projector_generic = indirect_projector_generic
	lib.indirect_projector_generic_p = POINTER(lib.indirect_projector_generic)

	lib.indirect_projector_generic_alloc.argtypes = [operator_p]
	lib.indirect_projector_generic_alloc.restype = projector_p
