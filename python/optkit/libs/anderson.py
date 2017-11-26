from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import include_ok_dense, ok_dense_API

def include_ok_anderson(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'anderson_accelerator_p', attach_anderson_ctypes, **include_args)

def ok_anderson_API(): return ok_dense_API() + [attach_anderson_ccalls]

class AndersonLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libanderson_', ok_anderson_API())

def attach_anderson_ctypes(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	class anderson_accelerator(ct.Structure):
		_fields_ = [('vector_dim', ct.c_size_t),
					('lookback_dim', ct.c_size_t),
					('F', lib.matrix_p),
					('G', lib.matrix_p),
					('F_gram', lib.matrix_p),
					('f', lib.vector_p),
					('g', lib.vector_p),
					('diag', lib.vector_p),
					('alpha', lib.vector_p),
					('ones', lib.vector_p),
					('mu_regularization', lib.ok_float),
					('iter', ct.c_size_t),
					('linalg_handle', ct.c_void_p)]

	lib.anderson_accelerator = anderson_accelerator
	lib.anderson_accelerator_p = ct.POINTER(lib.anderson_accelerator)

def attach_anderson_ccalls(lib, single_precision=False):
	include_ok_anderson(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p
	anderson_accelerator_p = lib.anderson_accelerator_p

	# argument types
	lib.anderson_accelerator_init.argtypes = [
			anderson_accelerator_p, ct.c_size_t, ct.c_size_t]
	lib.anderson_accelerator_free.argtypes = [anderson_accelerator_p]
	lib.anderson_set_x0.argtypes = [
			anderson_accelerator_p, vector_p]
	lib.anderson_update_F_x.argtypes = [
			anderson_accelerator_p, matrix_p, vector_p, ct.c_size_t]
	lib.anderson_update_F_g.argtypes = [
			anderson_accelerator_p, matrix_p, vector_p, ct.c_size_t]
	lib.anderson_update_G.argtypes = [
			anderson_accelerator_p, matrix_p, vector_p, ct.c_size_t]
	lib.anderson_regularized_gram.argtypes = [
			anderson_accelerator_p, matrix_p, matrix_p, ok_float]
	lib.anderson_solve.argtypes = [
			anderson_accelerator_p, matrix_p, vector_p, ok_float]
	lib.anderson_mix.argtypes = [
			anderson_accelerator_p, matrix_p, vector_p, vector_p]
	lib.anderson_accelerate.argtypes = [anderson_accelerator_p, vector_p]

	# return types
	OptkitLibs.attach_default_restype(
			lib.anderson_accelerator_init,
			lib.anderson_accelerator_free,
			lib.anderson_set_x0,
			lib.anderson_update_F_x,
			lib.anderson_update_F_g,
			lib.anderson_update_G,
			lib.anderson_regularized_gram,
			lib.anderson_solve,
			lib.anderson_mix,
			lib.anderson_accelerate,
	)