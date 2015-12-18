from numpy import nan

class ProxKernels(object):

	def __init__(self, backend, vector_type, function_vector_type):
		self.on_gpu = backend.device == 'gpu'
		self.float = backend.lowtypes.FLOAT_CAST
		self.proxlib = backend.prox
		self.ndarray_pointer = backend.lowtypes.ndarray_pointer
		self.Vector = vector_type
		self.FunctionVector = function_vector_type

		self.dimcheck = backend.dimcheck
		self.typecheck = backend.typecheck
		self.devicecheck = backend.devicecheck


	def device_compare(self, *args):
		for arg in args:
			if arg.on_gpu != self.on_gpu:
				raise ValueError("kernel call on GPU ={}\n"
					"all inputs on GPU = {}".format(self.on_gpu, arg.on_gpu))

	def scale_function_vector(self, f, v, mul):
		if not isinstance(f, self.FunctionVector):
			raise TypeError("optkit.FunctionVector required "
							"as first argument.\n Provided: {}\n".format(
									type(f)))

		if not isinstance(v, self.Vector):
			raise TypeError("optkit.Vector required "
							"as second argument.\n Provided: {}\n".format(
								type(v)))

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
				raise TypeError("optkit.FunctionVector required.")

			if self.devicecheck: self.device_compare(f)


			self.proxlib.function_vector_memcpy_vmulti(f.c,
					self.ndarray_pointer(f.h_, function = True),
					self.ndarray_pointer(f.a_, function = False),
					self.ndarray_pointer(f.b_, function = False),
					self.ndarray_pointer(f.c_, function = False),
					self.ndarray_pointer(f.d_, function = False),
					self.ndarray_pointer(f.e_, function = False))	

	def print_function_vector(self, f):
		if not isinstance(f, self.FunctionVector):
			raise TypeError("optkit.FunctionVector required.")

		if self.devicecheck: self.device_compare(f)

		self.proxlib.function_vector_print(f.c)

	def eval(self, f, x, typecheck=None, dimcheck=None):
		if self.typecheck:
			if not isinstance(f, self.FunctionVector):
				raise TypeError("`optkit.FunctionVector` required as "
					"first argument.")
			if not isinstance(x, self.Vector):
				raise TypeError("`optkit.Vector` required as second "
					"argument.")

		if self.dimcheck:
			if not f.size == x.size:
				raise TypeError("argument sizes incompatible"
						"size f: {}\nsize x: {}\n".format(f.size, x.size))
		
		if self.devicecheck: self.device_compare(f, x, x_out)

		return self.proxlib.FuncEvalVector(f.c, x.c)

	def prox(self, f, rho, x, x_out, typecheck=None, dimcheck=None):

		if self.typecheck:
			if not isinstance(f, self.FunctionVector):
				raise TypeError("`optkit.FunctionVector` required as "
					"first argument.")
			if not isinstance(rho, (int, float, self.float)):
				raise TypeError("`int` or `float` required as second argument.")
			if not isinstance(x, self.Vector):
				raise TypeError("`optkit.Vector` required as third "
					"argument.")
			if not isinstance(x_out, self.Vector):
				raise TypeError("`optkit.Vector` required as fourth "
					"argument")

		if self.dimcheck:
			if not f.size == x.size and f.size == x_out.size:
				raise ValueError("argument sizes incompatible:\n"
						"size f: {}\nsize x: {}\n size x_out: {}\n".format(
							f.size, x.size, x_out.size))

		if self.devicecheck: self.device_compare(f, x, x_out)

		self.proxlib.ProxEvalVector(f.c, rho, x.c, x_out.c)
