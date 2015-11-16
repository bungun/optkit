from optkit.types import ok_function_vector, ok_function_enums as fcn_enums
from optkit.libs import proxlib
from optkit.utils import ndarray_pointer
from numpy import ndarray

# low-level utilities
def make_cfunctionvector(f=None, copy_data=True):
	if f is None:
		return ok_function_vector(0,None)
	elif isinstance(f, ndarray) and \
		 len(f.shape)==1 and \
		 f.dtype == function_dt:
		f_ = ok_function_vector(0,None)
		if not copy_data:
			proxlib.function_vector_view_array(f_, 
				ndarray_pointer(f, function=True), f.size)
		else:
			proxlib.function_vector_calloc(f_, f.size)
			proxlib.function_vector_memcpy_va(f_, 
				ndarray_pointer(f.h, function = True),
				ndarray_pointer(f.a, function = False),
				ndarray_pointer(f.b, function = False),
				ndarray_pointer(f.c, function = False),
				ndarray_pointer(f.d, function = False),
				ndarray_pointer(f.e, function = False))	
		return f_
	else:
		return None
		# TODO: error message (type, dims)

def release_cfunctionvector(f):
	if isinstance(f, ok_function_vector):
		proxlib.function_vector_free(f)