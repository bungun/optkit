from ctypes import c_float, c_double, c_int, c_size_t, POINTER, Structure
from optkit.defs import FLOAT_FLAG

ok_float = c_float if FLOAT_FLAG else c_double

# pointers to C types
c_int_p = POINTER(c_int)
ok_float_p = POINTER(ok_float)

# enums
class OKEnums(object):
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

ok_enums = OKEnums()

# low-level optkit types
class ok_vector(Structure):
	_fields_ = [('size', c_size_t),('stride', c_size_t),('data', ok_float_p)]

class ok_matrix(Structure):
	_fields_ = [('size1',c_size_t),
				('size2',c_size_t),
				('tda',c_size_t),
				('data',ok_float_p),
				('rowmajor',c_int)]


# pointers to optkit types
vector_p = POINTER(ok_vector);
matrix_p = POINTER(ok_matrix)

