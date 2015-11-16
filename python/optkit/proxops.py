from optkit.libs import proxlib
from optkit.types import FunctionVector, Vector
from optkit.utils import ndarray_pointer
from numpy import nan

# TODO: raise exceptions instead of printing errors directly

def push_function_vector(f):
	if not isinstance(f, FunctionVector):
		print ("Error: optkit.FunctionVector required")

	proxlib.function_vector_memcpy_vmulti(f.c,
			ndarray_pointer(f.h_, function = True),
			ndarray_pointer(f.a_, function = False),
			ndarray_pointer(f.b_, function = False),
			ndarray_pointer(f.c_, function = False),
			ndarray_pointer(f.d_, function = False),
			ndarray_pointer(f.e_, function = False))	

def print_function_vector(f):
	if not isinstance(f, FunctionVector):
		print ("Error: optkit.FunctionVector required")

	if not f.c is None:
		proxlib.function_vector_print(f.c)
	else:
		print ("Uninitialized optkit.FunctionVector\n")

def eval(f, x):
	if not isinstance(f, FunctionVector):
		print ("Error: `optkit.FunctionVector` required as first argument")
		return nan 
	if not isinstance(x, Vector):
		print ("Error: `optkit.Vector` required as second argument")
		return nan
	if not f.size == x.size:
		print ("Error: argument sizes incompatible"
				"size f: {}\nsize x: {}\n".format(f.size, x.size))
		return nan
	if not f.c is None:
		return proxlib.FuncEvalVector(f.c, x.c)
	else:
		print ("Uninitialized optkit.FunctionVector\n")


def prox(f, rho, x, x_out):
	if not isinstance(f, FunctionVector):
		print ("Error: `optkit.FunctionVector` required as first argument")
		return
	if not isinstance(rho, (int,float)):
		print ("Error: `int` or `float` required as second argument")
		return
	if not isinstance(x, Vector):
		print ("Error: `optkit.Vector` required as third argument")
		return
	if not isinstance(x_out, Vector):
		print ("Error: `optkit.Vector` required as fourth argument")
		return
	if not f.size == x.size and f.size == x_out.size:
		print ("Error: argument sizes incompatible:\n"
				"size f: {}\nsize x: {}\n size x_out: {}\n".format(
					f.size, x.size, x_out.size))
		return

	if not f.c is None:
		proxlib.ProxEvalVector(f.c, rho, x.c, x_out.c)
	else:
		print ("Uninitialized optkit.FunctionVector\n")
