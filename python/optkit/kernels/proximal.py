from numpy import nan

class ProxKernels(object):

	def __init__(self, backend, vector_type, function_vector_type):
		self.proxlib = backend.prox
		self.ndarray_pointer = backend.lowtypes.ndarray_pointer
		self.Vector = vector_type
		self.FunctionVector = function_vector_type

		self.CHK_MSG = str("\nMake sure to not mix backends: device "
			"(CPU vs. GPU) and floating pointer precision (32- vs 64-bit) "
			"must match.\n Current settings: {}-bit precision, "
			"{}.".format(backend.precision, backend.device))

	def scale_function_vector(self, f, v, mul):
		
		if not isinstance(f, self.FunctionVector):
			raise TypeError("Error: optkit.FunctionVector required "
							"as first argument.\n Provided: {}\n{}".format(
									type(f), self.CHK_MSG))

		if not isinstance(v, self.Vector):
			raise TypeError("Error: optkit.Vector required "
							"as second argument.\n Provided: {}\n".format(
								type(v), self.CHK_MSG))
		if mul:
			f.a_ *= v.py
			f.d_ *= v.py
			f.e_ *= v.py
		else:
			f.a_ /= v.py
			f.d_ /= v.py
			f.e_ /= v.py


	def push_function_vector(self, *function_vectors):
		for f in function_vectors:
			if not isinstance(f, self.FunctionVector):
				raise TypeError("Error: optkit.FunctionVector required. {}".format(
					self.CHK_MSG))

			self.proxlib.function_vector_memcpy_vmulti(f.c,
					self.ndarray_pointer(f.h_, function = True),
					self.ndarray_pointer(f.a_, function = False),
					self.ndarray_pointer(f.b_, function = False),
					self.ndarray_pointer(f.c_, function = False),
					self.ndarray_pointer(f.d_, function = False),
					self.ndarray_pointer(f.e_, function = False))	

	def print_function_vector(self, f):
		if not isinstance(f, self.FunctionVector):
			raise TypeError("optkit.FunctionVector required. {}".format(
				self.CHK_MSG))

		if not f.c is None:
			self.proxlib.function_vector_print(f.c)
		else:
			raise ValueError("Uninitialized optkit.FunctionVector\n")

	def eval(self, f, x):
		if not isinstance(f, self.FunctionVector):
			raise TypeError("`optkit.FunctionVector` required as "
				"first argument. {}".format(self.CHK_MSG))
		if not isinstance(x, self.Vector):
			raise TypeError("`optkit.Vector` required as second "
				"argument. {}".format(self.CHK_MSG))
		if not f.size == x.size:
			raise TypeError("argument sizes incompatible"
					"size f: {}\nsize x: {}\n".format(f.size, x.size))
		if not f.c is None:
			return self.proxlib.FuncEvalVector(f.c, x.c)
		else:
			raise ValueError("Uninitialized optkit.FunctionVector\n")

	def prox(self, f, rho, x, x_out):
		if not isinstance(f, self.FunctionVector):
			raise TypeError("`optkit.FunctionVector` required as "
				"first argument. {}".format(self.CHK_MSG))
		if not isinstance(rho, (int,float)):
			raise TypeError("`int` or `float` required as second argument.")
		if not isinstance(x, self.Vector):
			raise TypeError("`optkit.Vector` required as third "
				"argument. {}".format(self.CHK_MSG))
			return
		if not isinstance(x_out, self.Vector):
			raise TypeError("`optkit.Vector` required as fourth "
				"argument".format(self.CHK_MSG))
			return
		if not f.size == x.size and f.size == x_out.size:
			raise ValueError("argument sizes incompatible:\n"
					"size f: {}\nsize x: {}\n size x_out: {}\n".format(
						f.size, x.size, x_out.size))
			return

		if not f.c is None:
			self.proxlib.ProxEvalVector(f.c, rho, x.c, x_out.c)
		else:
			raise ValueError("Uninitialized optkit.FunctionVector\n")
