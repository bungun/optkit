from optkit.types import enums, ok_float_p, ok_vector, ok_matrix
from optkit.defs import __float_conversion
from optkit.libs import oklib
from numpy import ndarray

# low-level utilities
def ndarray_pointer(x):
	if isinstance(x,ndarray):
		return __float_conversion(x).ctypes.data_as(ok_float_p)
	else:
		return None
		# TODO: error message? raise error?


def make_cvector(x=None):
	if x is None:
		return ok_vector(0,0,None)
	elif isinstance(x, ndarray) and len(x.shape)==1:
		x_ = ok_vector(0,0,None)
		oklib.__vector_view_array(x_, ndarray_pointer(x), x.size[0]) 
		return x
	else:
		return None
		# TODO: error message (type, dims)

def make_cmatrix(A=None):
	if A is None:
		return ok_matrix(0,0,0,None,enums.CblasRowMajor)
	elif isinstance(A, ndarray) and len(A.shape)==2:
		(m,n)=A.shape
		A_ = ok_matrix(0,0,0,None,enums.CblasRowMajor)
		oklib.__matrix_view_array(A_, ndarray_pointer(A), 
									m, n, enums.CblasRowMajor)
		return A_
	else:
		return None
		# TODO: error message (type, dims)

