from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_base_ccalls, attach_vector_ccalls, attach_dense_linsys_ccalls

class AndersonLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libanderson_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_dense_linsys_ccalls)
		self.attach_calls.append(attach_anderson_ctypes)
		self.attach_calls.append(attach_anderson_ccalls)

def attach_anderson_ctypes(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p

	class anderson_accelerator(ct.Structure):
		_fields_ = [('vector_dim', ct.c_size_t),
					('lookback_dim', ct.c_size_t),
					('F', matrix_p),
					('G', matrix_p),
					('F_gram', matrix_p),
					('f', vector_p),
					('g', vector_p),
					('diag', vector_p),
					('alpha', vector_p),
					('ones', vector_p),
					('mu_regularization', ok_float),
					('iter', ct.c_size_t),
					('linalg_handle', ct.c_void_p)]

	lib.anderson_accelerator = anderson_accelerator
	lib.anderson_accelerator_p = ct.POINTER(lib.anderson_accelerator)

def attach_anderson_ccalls(lib, single_precision=False):
	if not 'anderson_accelerator_p' in lib.__dict__:
		attach_anderson_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p
	anderson_accelerator_p = lib.anderson_accelerator_p

	# argument types
	lib.anderson_accelerator_init.argtypes = [vector_p, ct.c_size_t]
	lib.anderson_accelerator_free.argtypes = [anderson_accelerator_p]
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
	lib.anderson_accelerator_init.restype = anderson_accelerator_p
	lib.anderson_accelerator_free.restype = ct.c_uint
	lib.anderson_update_F_x.restype = ct.c_uint
	lib.anderson_update_F_g.restype = ct.c_uint
	lib.anderson_update_G.restype = ct.c_uint
	lib.anderson_regularized_gram.restype = ct.c_uint
	lib.anderson_solve.restype = ct.c_uint
	lib.anderson_mix.restype = ct.c_uint
	lib.anderson_accelerate.restype = ct.c_uint