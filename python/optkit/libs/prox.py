from ctypes import Structure, POINTER, c_int, c_uint, c_size_t, c_void_p
from numpy import float32
from site import getsitepackages
from optkit.libs.loader import OptkitLibs
from optkit.libs.enums import OKFunctionEnums
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_sparse_linsys_ctypes, attach_base_ccalls, attach_vector_ccalls, \
	attach_dense_linsys_ccalls, attach_sparse_linsys_ccalls

class ProxLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libprox_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_prox_ctypes)
		self.attach_calls.append(attach_prox_ccalls)

		self.function_enums = OKFunctionEnums()

def attach_prox_ctypes(lib, single_precision=False):
	if 'ok_float' not in lib.__dict__:
		attach_base_ctypes(lib, single_precision)

	ok_float = lib.ok_float

	# function struct
	class ok_function(Structure):
		_fields_ = [('h', c_uint),
					('a', ok_float),
					('b', ok_float),
					('c', ok_float),
					('d', ok_float),
					('e', ok_float)]
	lib.function = ok_function
	lib.function_p = POINTER(lib.function)

	# function vector struct
	class ok_function_vector(Structure):
		_fields_ = [('size', c_size_t),
					('objectives', function_p)]
	lib.function_vector = ok_function_vector
	lib.function_vector_p = POINTER(lib.function_vector)

def attach_prox_ccalls(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'function_vector_p' in lib.__dict__:
		attach_prox_ctypes(lib, single_precision)

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
	lib.function_vector_alloc.restype = None
	lib.function_vector_calloc.restype = None
	lib.function_vector_free.restype = None
	lib.function_vector_memcpy_va.restype = None
	lib.function_vector_memcpy_av.restype = None
	lib.function_vector_mul.restype = None
	lib.function_vector_div.restype = None
	lib.function_vector_print.restype = None

	# Prox & Function evaluation
	# --------------------------
	## arguments
	lib.ProxEvalVector.argtypes = [function_vector_p, ok_float, vector_p,
								   vector_p]
	lib.FuncEvalVector.argtypes = [function_vector_p, vector_p]

	## return values
	lib.ProxEvalVector.restype = None
	lib.FuncEvalVector.restype = ok_float