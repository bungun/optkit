from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_sparse_linsys_ctypes, attach_base_ccalls, attach_vector_ccalls, \
	attach_dense_linsys_ccalls, attach_sparse_linsys_ccalls
from optkit.libs.operator import attach_operator_ctypes, attach_operator_ccalls

class EquilibrationLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libequil_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_sparse_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_dense_linsys_ccalls)
		self.attach_calls.append(attach_sparse_linsys_ccalls)
		self.attach_calls.append(attach_operator_ctypes)
		self.attach_calls.append(attach_operator_ccalls)
		self.attach_calls.append(attach_equilibration_ccalls)
		self.attach_calls.append(attach_operator_equilibration_ccalls)

def attach_equilibration_ccalls(lib, single_precision=False):
	if not 'matrix_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)

	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p

	# argument types
	lib.regularized_sinkhorn_knopp.argtypes = [
			ct.c_void_p, ok_float_p, matrix_p, vector_p, vector_p, ct.c_uint]

	# return types
	lib.regularized_sinkhorn_knopp.restype = ct.c_uint

def attach_operator_equilibration_ccalls(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'operator_p' in lib.__dict__:
		attach_operator_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p
	operator_p = lib.operator_p

	# argument types
	lib.operator_regularized_sinkhorn.argtypes = [
			ct.c_void_p, operator_p, vector_p, vector_p, ok_float]
	lib.operator_equilibrate.argtypes = [
			ct.c_void_p, operator_p, vector_p, vector_p, ok_float]
	lib.operator_estimate_norm.argtypes = [ct.c_void_p, operator_p]

	# return types
	lib.operator_regularized_sinkhorn.restype = ct.c_uint
	lib.operator_equilibrate.restype = ct.c_uint
	lib.operator_estimate_norm.restype = ct.c_uint