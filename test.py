from ctypes import *



gsl = CDLL('/Users/Baris/Documents/Thesis/modules/gsl_c/build/libgsl.dylib')

# pointers to C types
c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)
c_double_p = POINTER(c_double)

class vector(Structure):
	_fields_ = [('size', c_size_t),('stride', c_size_t),('data', c_double_p)]

def make_vector():
	return vector(0,0,None)

vector_p = POINTER(vector);

v = make_vector()
w = make_vector()


gsl.__vector_alloc.argtypes=[vector_p, c_size_t]
gsl.__vector_free.argtypes=[vector_p]
gsl.__vector_print.argtypes=[vector_p]
gsl.__vector_add_constant.argtypes=[vector_p, c_double]

gsl.__vector_alloc.restype=None
gsl.__vector_free.restype=None
gsl.__vector_print.restype=None
gsl.__vector_add_constant.restype=None


gsl.__vector_alloc(v,10)
gsl.__vector_add_constant(v,12.)
gsl.__vector_print(v)
gsl.__vector_free(v)



