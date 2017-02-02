from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_sparse_linsys_ctypes, attach_base_ccalls, attach_vector_ccalls, \
	attach_dense_linsys_ccalls, attach_sparse_linsys_ccalls
from optkit.libs.operator import attach_operator_ctypes, attach_operator_ccalls

class ConjugateGradientLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libcg_')
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

def attach_cg_ctypes(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p

	class cgls_helper(ct.Structure):
		_fields_ = [('p', vector_p),
					('q', vector_p),
					('r', vector_p),
					('s', vector_p),
					('norm_s', ok_float),
					('norm_s0', ok_float),
					('norm_x', ok_float),
					('xmax', ok_float),
					('alpha', ok_float),
					('beta', ok_float),
					('delta', ok_float),
					('gamma', ok_float),
					('gamma_prev', ok_float),
					('shrink', ok_float),
					('blas_handle', ct.c_void_p)]

	lib.cgls_helper = cgls_helper
	lib.cgls_helper_p = ct.POINTER(lib.cgls_helper)

	class pcg_helper(ct.Structure):
		_fields_ = [('p', vector_p),
					('q', vector_p),
					('r', vector_p),
					('z', vector_p),
					('temp', vector_p),
					('norm_r', ok_float),
					('alpha', ok_float),
					('gamma', ok_float),
					('gamma_prev', ok_float),
					('blas_handle', ct.c_void_p),
					('never_solved', ct.c_int)]

	lib.pcg_helper = pcg_helper
	lib.pcg_helper_p = ct.POINTER(lib.pcg_helper)

def attach_cg_ccalls(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'operator_p' in lib.__dict__:
		attach_operator_ctypes(lib, single_precision)
	if not 'cgls_helper_p' in lib.__dict__:
		attach_cg_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p
	operator_p = lib.operator_p
	cgls_helper_p = lib.cgls_helper_p
	pcg_helper_p = lib.pcg_helper_p

	c_uint_p = ct.POINTER(ct.c_uint)

	# argument types
	lib.cgls_helper_alloc.argtypes = [ct.c_size_t, ct.c_size_t]
	lib.cgls_helper_free.argtypes = [cgls_helper_p]

	lib.cgls_nonallocating.argtypes = [
			cgls_helper_p, operator_p, vector_p, vector_p, ok_float, ok_float,
			ct.c_size_t, ct.c_int, c_uint_p]
	lib.cgls.argtypes = [
			operator_p, vector_p, vector_p, ok_float, ok_float, ct.c_size_t,
			ct.c_int, c_uint_p]
	lib.cgls_init.argtypes = [ct.c_size_t, ct.c_size_t]
	lib.cgls_solve.argtypes = [
			ct.c_void_p, operator_p, vector_p, vector_p, ok_float, ok_float,
			ct.c_size_t, ct.c_int, c_uint_p]
	lib.cgls_finish.argtypes = [ct.c_void_p]
	lib.CGLS_MAXFLAG = 4;

	lib.pcg_helper_alloc.argtypes = [ct.c_size_t, ct.c_size_t]
	lib.pcg_helper_free.argtypes = [pcg_helper_p]

	lib.diagonal_preconditioner.argtypes = [
			operator_p, vector_p, ok_float]

	lib.pcg_nonallocating.argtypes = [
			pcg_helper_p, operator_p, operator_p, vector_p, vector_p, ok_float,
			ok_float, ct.c_size_t, ct.c_int, c_uint_p]
	lib.pcg.argtypes = [
			operator_p, operator_p, vector_p, vector_p, ok_float, ok_float,
			ct.c_size_t, ct.c_int, c_uint_p]
	lib.pcg_init.argtypes = [ct.c_size_t, ct.c_size_t]
	lib.pcg_solve.argtypes = [
			ct.c_void_p, operator_p, operator_p, vector_p, vector_p, ok_float,
			ok_float, ct.c_size_t, ct.c_int, c_uint_p]
	lib.pcg_finish.argtypes = [ct.c_void_p]

	# return types
	lib.cgls_helper_alloc.restype = cgls_helper_p
	lib.cgls_helper_free.retype = ct.c_uint

	lib.cgls_nonallocating.restype = ct.c_uint
	lib.cgls.restype = ct.c_uint
	lib.cgls_init.restype = ct.c_void_p
	lib.cgls_solve.restype = ct.c_uint
	lib.cgls_finish.restype = ct.c_uint

	lib.pcg_helper_alloc.restype = pcg_helper_p
	lib.pcg_helper_free.retype = ct.c_uint

	lib.diagonal_preconditioner.restype = ct.c_uint

	lib.pcg_nonallocating.restype = ct.c_uint
	lib.pcg.restype = ct.c_uint
	lib.pcg_init.restype = ct.c_void_p
	lib.pcg_solve.restype = ct.c_uint
	lib.pcg_finish.restype = ct.c_uint