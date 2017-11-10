from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import include_ok_dense, ok_linsys_api
from optkit.libs.operator import include_ok_operator
from optkit.libs.cg import ok_cg_api

def include_ok_projector(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'projector_p', attach_projector_ctypes, **include_args)

ok_projector_dense_api = ok_dense_api + [attach_projector_ccalls]
ok_projector_api = ok_cg_api + [
		attach_projector_ccalls, attach_operator_projector_ctypes_ccalls]

class ProjectorLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libprojector_', ok_projector_api)

def attach_projector_ctypes(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p

	class direct_projector(ct.Structure):
		_fields_ = [('A', matrix_p),
					('L', matrix_p),
					('normA', ok_float),
					('skinny', ct.c_int),
					('normalized', ct.c_int)]

	lib.direct_projector = direct_projector
	lib.direct_projector_p = ct.POINTER(lib.direct_projector)

	class projector(ct.Structure):
		_fields_ = [('kind', ct.c_uint),
					('size1', ct.c_size_t),
					('size2', ct.c_size_t),
					('data', ct.c_void_p),
					('initialize', ct.CFUNCTYPE(
							ct.c_uint, ct.c_void_p, ct.c_int)),
					('project', ct.CFUNCTYPE(
							ct.c_uint, ct.c_void_p, vector_p, vector_p,
							vector_p, vector_p, ok_float)),
					('free', ct.CFUNCTYPE(ct.c_uint, ct.c_void_p))]

	lib.projector = projector
	lib.projector_p = ct.POINTER(lib.projector)

	class dense_direct_projector(ct.Structure):
		_fields_ = [('A', matrix_p),
					('L', matrix_p),
					('linalg_handle', ct.c_void_p),
					('normA', ok_float),
					('skinny', ct.c_int),
					('normalized', ct.c_int)]

	lib.dense_direct_projector = dense_direct_projector
	lib.dense_direct_projector_p = ct.POINTER(lib.dense_direct_projector)

def attach_projector_ccalls(lib, single_precision=False):
	include_ok_projector(lib, single_precision=single_precision)

	vector_p = lib.vector_p
	matrix_p = lib.matrix_p
	direct_projector_p = lib.direct_projector_p
	projector_p = lib.projector_p

	# args:
	# -direct
	lib.direct_projector_alloc.argtypes = [direct_projector_p, matrix_p]
	lib.direct_projector_initialize.argtypes = [
			ct.c_void_p, direct_projector_p, ct.c_int]
	lib.direct_projector_project.argtypes = [
			ct.c_void_p, direct_projector_p, vector_p, vector_p, vector_p,
			vector_p]
	lib.direct_projector_free.argtypes = [direct_projector_p]
	lib.dense_direct_projector_alloc.argtypes = [matrix_p]

	# returns:
	# -direct
	OptkitLibs.attach_default_restype([
			lib.direct_projector_alloc,
			lib.direct_projector_initialize,
			lib.direct_projector_project,
			lib.direct_projector_free,
	])
	lib.dense_direct_projector_alloc.restype = projector_p

def attach_operator_projector_ctypes_ccalls(lib, single_precision=False):
	include_ok_operator(lib, single_precision=single_precision)
	include_ok_projector(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p
	operator_p = lib.operator_p
	projector_p = lib.projector_p

	# types
	class indirect_projector(ct.Structure):
		_fields_ = [('A', operator_p),
					('cgls_work', ct.c_void_p),
					('flag', ct.c_uint)]

	lib.indirect_projector = indirect_projector
	lib.indirect_projector_p = ct.POINTER(lib.indirect_projector)
	indirect_projector_p = lib.indirect_projector_p

	class indirect_projector_generic(ct.Structure):
		_fields_ = [('A', operator_p),
					('cgls_work', ct.c_void_p),
					('linalg_handle', ct.c_void_p),
					('normA', ok_float),
					('normalized', ct.c_int),
					('flag', ct.c_uint)]

	lib.indirect_projector_generic = indirect_projector_generic
	lib.indirect_projector_generic_p = ct.POINTER(
			lib.indirect_projector_generic)

	# calls
	lib.indirect_projector_alloc.argtypes = [indirect_projector_p, operator_p]
	lib.indirect_projector_initialize.argtypes = [
			ct.c_void_p, indirect_projector_p, ct.c_int]
	lib.indirect_projector_project.argtypes = [
			ct.c_void_p, indirect_projector_p, vector_p, vector_p, vector_p,
			vector_p]
	lib.indirect_projector_free.argtypes = [indirect_projector_p]
	lib.indirect_projector_generic_alloc.argtypes = [operator_p]

	OptkitLibs.attach_default_restype([
			lib.indirect_projector_alloc,
			lib.indirect_projector_initialize,
			lib.indirect_projector_project,
			lib.indirect_projector_free,
	])
	lib.indirect_projector_generic_alloc.restype = projector_p
