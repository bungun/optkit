from optkit.types import ok_enums as enums, ok_float_p, \
					     ok_vector, ok_matrix
from optkit.defs import FLOAT_CAST
from optkit.libs import oklib
from optkit.utils import ndarray_pointer
from numpy import ndarray

# low-level utilities
def make_cvector(x=None, copy_data=True):
	if x is None:
		return ok_vector(0,0,None)
	elif isinstance(x, ndarray) and len(x.shape)==1:
		x_ = ok_vector(0,0,None)
		if not copy_data:
			oklib.__vector_view_array(x_, ndarray_pointer(x), x.size)
		else:
			oklib.__vector_calloc(x_, x.size)
			oklib.__vector_memcpy_va(x_, ndarray_pointer(x))	 
		return x_
	else:
		return None
		# TODO: error message (type, dims)


def make_cmatrix(A=None, copy_data=True):

	if A is None:
		return ok_matrix(0,0,0,None,enums.CblasRowMajor)
	elif isinstance(A, ndarray) and len(A.shape)==2:
		# ASSUMES DATA IS ROW-MAJOR
		(m,n) = A.shape
		A_ = ok_matrix(0,0,0,None,enums.CblasRowMajor)
		if not copy_data:
			oklib.__matrix_view_array(A_, ndarray_pointer(A), 
										m, n, enums.CblasRowMajor)
		else:
			oklib.__matrix_calloc(A_, m, n, enums.CblasRowMajor)
			oklib.__matrix_memcpy_ma(A_, ndarray_pointer(A))
		return A_
	else:
		return None
		# TODO: error message (type, dims)


def release_cvector(x):
	if isinstance(x, ok_vector):
		oklib.__vector_free(x)

def release_cmatrix(A):
	if isinstance(A, ok_matrix):
		oklib.__matrix_free(x)


