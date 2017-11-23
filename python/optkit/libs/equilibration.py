from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import include_ok_dense, ok_dense_API
from optkit.libs.operator import include_ok_operator, ok_operator_API

def ok_equil_dense_API(): return ok_dense_API() + [attach_equilibration_ccalls]
def ok_equil_API(): return (
		ok_operator_API()
		+ [attach_equilibration_ccalls, attach_operator_equilibration_ccalls]
	)

class EquilibrationLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libequil_', ok_equil_generic_API())

def attach_equilibration_ccalls(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p

	# argument types
	lib.regularized_sinkhorn_knopp.argtypes = [
			ct.c_void_p, ok_float_p, matrix_p, vector_p, vector_p, ct.c_uint]

	# return types
	OptkitLibs.attach_default_restype([lib.regularized_sinkhorn_knopp])

def attach_operator_equilibration_ccalls(lib, single_precision=False):
	include_ok_operator(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p
	abstract_operator_p = lib.abstract_operator_p

	# argument types
	lib.operator_regularized_sinkhorn.argtypes = [
			ct.c_void_p, abstract_operator_p, vector_p, vector_p, ok_float]
	lib.operator_equilibrate.argtypes = [
			ct.c_void_p, abstract_operator_p, vector_p, vector_p, ok_float]
	lib.operator_estimate_norm.argtypes = [ct.c_void_p, abstract_operator_p]

	# return types
	OptkitLibs.attach_default_restype([
			lib.operator_regularized_sinkhorn,
			lib.operator_equilibrate,
			lib.operator_estimate_norm,
	])