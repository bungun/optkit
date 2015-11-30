from optkit.libs import proxlib
from optkit.types import FunctionVector, Vector
from optkit.utils import ndarray_pointer
from numpy import nan

# TODO: raise exceptions instead of printing errors directly


  # // Scale f and g to account for diagonal scaling e and d.
  # for (unsigned int i = 0; i < m && !err; ++i) {
  #   f[i].a /= gsl::vector_get(&d, i);
  #   f[i].d /= gsl::vector_get(&d, i);
  # }
  # for (unsigned int j = 0; j < n && !err; ++j) {
  #   g[j].a *= gsl::vector_get(&e, j);
  #   g[j].d *= gsl::vector_get(&e, j);
  # }


def scale_function_vector(f, v, mul=True):
	if not isinstance(f, FunctionVector):
		raise TypeError("Error: optkit.FunctionVector required "
						"as first argument.\n Provided: {}\n".format(
								type(f)))
#
	if not isinstance(v, Vector):
		raise TypeError("Error: optkit.Vector required "
						"as second argument.\n Provided: {}\n".format(
							type(v)))
#
	if mul:
		f.a_ *= v.py
		f.d_ *= v.py
		f.e_ *= v.py
	else:
		f.a_ /= v.py
		f.d_ /= v.py
		f.e_ /= v.py




def push_function_vector(*function_vectors):
	for f in function_vectors:
		if not isinstance(f, FunctionVector):
			raise TypeError("Error: optkit.FunctionVector required")

		proxlib.function_vector_memcpy_vmulti(f.c,
				ndarray_pointer(f.h_, function = True),
				ndarray_pointer(f.a_, function = False),
				ndarray_pointer(f.b_, function = False),
				ndarray_pointer(f.c_, function = False),
				ndarray_pointer(f.d_, function = False),
				ndarray_pointer(f.e_, function = False))	

def print_function_vector(f):
	if not isinstance(f, FunctionVector):
		raise TypeError("optkit.FunctionVector required")

	if not f.c is None:
		proxlib.function_vector_print(f.c)
	else:
		raise ValueError("Uninitialized optkit.FunctionVector\n")

def eval(f, x):
	if not isinstance(f, FunctionVector):
		raise TypeError("`optkit.FunctionVector` required as first argument")
	if not isinstance(x, Vector):
		raise TypeError("`optkit.Vector` required as second argument")
	if not f.size == x.size:
		raise TypeError("argument sizes incompatible"
				"size f: {}\nsize x: {}\n".format(f.size, x.size))
	if not f.c is None:
		return proxlib.FuncEvalVector(f.c, x.c)
	else:
		raise ValueError("Uninitialized optkit.FunctionVector\n")


def prox(f, rho, x, x_out):
	if not isinstance(f, FunctionVector):
		raise TypeError("`optkit.FunctionVector` required as first argument")
	if not isinstance(rho, (int,float)):
		raise TypeError("`int` or `float` required as second argument")
	if not isinstance(x, Vector):
		raise TypeError("`optkit.Vector` required as third argument")
		return
	if not isinstance(x_out, Vector):
		raise TypeError("`optkit.Vector` required as fourth argument")
		return
	if not f.size == x.size and f.size == x_out.size:
		raise ValueError("argument sizes incompatible:\n"
				"size f: {}\nsize x: {}\n size x_out: {}\n".format(
					f.size, x.size, x_out.size))
		return

	if not f.c is None:
		proxlib.ProxEvalVector(f.c, rho, x.c, x_out.c)
	else:
		raise ValueError("Uninitialized optkit.FunctionVector\n")
