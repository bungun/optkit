from ctypes import c_float, c_double, c_int, c_size_t, POINTER, Structure
from optkit.defs import __float_flag, __gpu_flag
from optkit.utils import ndarray_pointer

ok_float = c_float if __float_flag else c_double

# pointers to C types
c_int_p = POINTER(c_int)
ok_float_p = POINTER(ok_float)

# enums
class ok_enums(object):
	CblasRowMajor = c_int(101)
	CblasColMajor = c_int(102)
	CblasNoTrans = c_int(111)
	CblasTrans = c_int(112)
	CblasConjTrans = c_int(113)
	CblasUpper = c_int(121)
	CblasLower = c_int(122)
	CblasNonUnit = c_int(131)
	CblasUnit = c_int(132)
	CblasLeft = c_int(141)
	CblasRight = c_int(142)
	def __init__(self):
		pass

enums = ok_enums()


# low-level optkit types
class ok_vector(Structure):
	_fields_ = [('size', c_size_t),('stride', c_size_t),('data', ok_float_p)]

class ok_matrix(Structure):
	_fields_ = [('size1',c_size_t),
				('size2',c_size_t),
				('tda',c_size_t),
				('data',ok_float_p),
				('rowmajor',c_int)]

class Vector(object):
	def __init__(self, x):
		self.local=x
		self.remote=ndarray_pointer(x)
		self.access_forbidden = __gpu_flag


# high-level optkit-python types
class Matrix(object):
	def __init__(self, A):
		self.local=A
		self.remote=ndarray_pointer(x)
		self.access_forbidden = __gpu_flag
		


# pointers to optkit types
vector_p = POINTER(ok_vector);
matrix_p = POINTER(ok_matrix)