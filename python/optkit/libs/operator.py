from ctypes import c_uint, c_size_t, c_void_p, CFUNCTYPE, POINTER, Structure
from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_sparse_linsys_ctypes, attach_base_ccalls, attach_vector_ccalls, \
	attach_dense_linsys_ccalls, attach_sparse_linsys_ccalls

class OperatorLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'liboperator_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_sparse_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_dense_linsys_ccalls)
		self.attach_calls.append(attach_sparse_linsys_ccalls)
		self.attach_calls.append(attach_operator_ctypes)
		self.attach_calls.append(attach_operator_ccalls)

def attach_operator_ctypes(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p

	class ok_operator(Structure):
		_fields_ = [('size1', c_size_t),
					('size2', c_size_t),
					('data', c_void_p),
					('apply', CFUNCTYPE(None, c_void_p, vector_p, vector_p)),
					('adjoint', CFUNCTYPE(None, c_void_p,vector_p, vector_p)),
					('fused_apply', CFUNCTYPE(None, c_void_p, ok_float,
											  vector_p, ok_float, vector_p)),
					('fused_adjoint', CFUNCTYPE(None, c_void_p, ok_float,
												vector_p, ok_float, vector_p)),
					('free', CFUNCTYPE(None, c_void_p)),
					('kind', c_uint)]

	lib.operator = ok_operator
	lib.operator_p = POINTER(lib.operator)

def attach_operator_ccalls(lib, single_precision=False):
	if not 'sparse_matrix_p' in lib.__dict__:
		attach_sparse_linsys_ctypes(lib, single_precision)
	if not 'operator_p' in lib.__dict__:
		attach_operator_ctypes(lib, single_precision)

	# check VECTORP, MATRIXP, SPARSEMATRIXP, OPERATORP exist
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p
	sparse_matrix_p = lib.sparse_matrix_p
	operator_p = lib.operator_p

	# argument types
	lib.dense_operator_alloc.argtypes = [matrix_p]
	lib.sparse_operator_alloc.argtypes = [sparse_matrix_p]
	lib.diagonal_operator_alloc.argtypes = [vector_p]

	# return types
	lib.dense_operator_alloc.restype = operator_p
	lib.sparse_operator_alloc.restype = operator_p
	lib.diagonal_operator_alloc.restype = operator_p