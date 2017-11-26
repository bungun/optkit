from optkit.compat import *

from ctypes import Structure, POINTER, c_int, c_uint, c_size_t, c_void_p
from numpy import float32

from optkit.libs.loader import OptkitLibs
from optkit.libs.enums import OKFunctionEnums
from optkit.libs.linsys import include_ok_dense, ok_vector_API

def include_ok_prox(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'function_p', attach_prox_ctypes, **include_args)

def ok_prox_API(): return ok_vector_API() + [attach_prox_ccalls]

class ProxLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libprox_', ok_prox_API())

def attach_prox_ctypes(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	# function struct
	class ok_function(Structure):
		_fields_ = [('h', c_uint),
					('a', lib.ok_float),
					('b', lib.ok_float),
					('c', lib.ok_float),
					('d', lib.ok_float),
					('e', lib.ok_float),
					('s', lib.ok_float)]
	lib.function = ok_function
	lib.function_p = POINTER(lib.function)

	# function vector struct
	class ok_function_vector(Structure):
		_fields_ = [('size', c_size_t),
					('objectives', lib.function_p)]
	lib.function_vector = ok_function_vector
	lib.function_vector_p = POINTER(lib.function_vector)

	lib.function_enums = OKFunctionEnums()

def attach_prox_ccalls(lib, single_precision=False):
	include_ok_prox(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	function_p = lib.function_p
	function_vector_p = lib.function_vector_p

	# Function Vector
	# ------
	## arguments
	lib.function_vector_alloc.argtypes = [function_vector_p, c_size_t]
	lib.function_vector_calloc.argtypes = [function_vector_p, c_size_t]
	lib.function_vector_free.argtypes = [function_vector_p]
	lib.function_vector_memcpy_va.argtypes = [function_vector_p, function_p]
	lib.function_vector_memcpy_av.argtypes = [function_p, function_vector_p]
	lib.function_vector_mul.argtypes = [function_vector_p, vector_p]
	lib.function_vector_div.argtypes = [function_vector_p, vector_p]
	lib.function_vector_print.argtypes = [function_vector_p]

	## return values
	OptkitLibs.attach_default_restype(
			lib.function_vector_alloc,
			lib.function_vector_calloc,
			lib.function_vector_free,
			lib.function_vector_memcpy_va,
			lib.function_vector_memcpy_av,
			lib.function_vector_mul,
			lib.function_vector_div,
			lib.function_vector_print
	)

	# Prox & Function evaluation
	# --------------------------
	## arguments
	lib.prox_eval_vector.argtypes = [function_vector_p, ok_float, vector_p,
									 vector_p]
	lib.function_eval_vector.argtypes = [function_vector_p, vector_p,
										 ok_float_p]

	## return values
	OptkitLibs.attach_default_restype(
			lib.prox_eval_vector,
			lib.function_eval_vector,
	)