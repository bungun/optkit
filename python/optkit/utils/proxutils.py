from optkit.types import ok_function_vector, ok_function_enums as fcn_enums
from optkit.libs import proxlib
from optkit.utils import ndarray_pointer
# from numpy import ndarray

# low-level utilities
def make_cfunctionvector(n=None):
	if n is None:
		return ok_function_vector(0,None)
	elif isinstance(n, int):
		f_ = ok_function_vector(0,None)
		proxlib.function_vector_calloc(f_, n)
		return f_
	else:
		return None
		# TODO: error message (type, dims)

def release_cfunctionvector(f):
	if isinstance(f, ok_function_vector):
		proxlib.function_vector_free(f)


