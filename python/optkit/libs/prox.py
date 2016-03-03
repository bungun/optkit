from ctypes import Structure, POINTER, c_int, c_uint, c_size_t, c_void_p
from numpy import float32
from site import getsitepackages
from optkit.libs.loader import retrieve_libs, validate_lib

class OKFunctionEnums(object):
	Zero = c_uint(0).value
	Abs = c_uint(1).value
	Exp = c_uint(2).value
	Huber = c_uint(3).value
	Identity = c_uint(4).value
	IndBox01 = c_uint(5).value
	IndEq0 = c_uint(6).value
	IndGe0 = c_uint(7).value
	IndLe0 = c_uint(8).value
	Logistic = c_uint(9).value
	MaxNeg0 = c_uint(10).value
	MaxPos0 = c_uint(11).value
	NegEntr = c_uint(12).value
	NegLog = c_uint(13).value
	Recipr = c_uint(14).value
	Square = c_uint(15).value
	__str2fun = {'Zero': 0, 'Abs': 1, 'Exp': 2, 'Huber': 3,
			'Identity': 4, 'IndBox01': 5, 'IndEq0': 6, 'IndGe0': 7,
			'IndLe0': 8, 'Logistic': 9, 'MaxNeg0': 10, 'MaxPos0': 11,
			'NegEntr': 12, 'NegLog': 13, 'Recipr': 14, 'Square': 15}
	min_enum = 0
	max_enum = max(__str2fun.values())

	@property
	def dict(self):
		return self.__str2fun

	def validate(self, h):
		if isinstance(h, int):
			if h < self.min_enum or h > self.max_enum:
				IndexError('value out of range: {}, '
							'(valid = {} to {}\n'.format(h,
							self.min_enum, self.max_enum))
			else:
				return c_uint(h).value
		elif isinstance(h, str):
			if not h in self.__str2fun:
				KeyError('invalid key: {}. valid keys:\n{}\n'.format(h,
					   self.__str2fun.keys()))
				return c_uint(0).value
			else:
				return c_uint(self.dict[h]).value
		else:
			TypeError('optkit.types.Function, field "h" can be initialized '
				'with arguments of type:\n "int", "c_int", or "str"\n')

	def validate_ce(self, c_or_e):
		if c_or_e < 0:
			ValueError('Function parameters "c" and "e" must be strictly '
				'positive for function to be convex:\n'
				'f(x) =def= c * h(ax - b) + dx + ex^2\n, with h convex.')
		else:
			return c_or_e

class ProxLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libprox_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, denselib, single_precision=False, gpu=False):
		device = 'gpu' if gpu else 'cpu'
		precision = '32' if single_precision else '64'
		lib_key = '{}{}'.format(device, precision)
		if lib_key not in self.libs:
			return None
		elif self.libs[lib_key] is None:
			return None

		lib = self.libs[lib_key]
		if lib.INITIALIZED:
			return lib
		else:
			ok_float = denselib.ok_float
			ok_float_p = denselib.ok_float_p
			vector_p = denselib.vector_p
			lib.enums = OKFunctionEnums()

			# function struct
			class ok_function(Structure):
				_fields_ = [('h', c_uint),
							('a', ok_float),
							('b', ok_float),
							('c', ok_float),
							('d', ok_float),
							('e', ok_float)]
			function_p = POINTER(ok_function)
			lib.function = ok_function
			lib.function_p = function_p

			# function vector struct
			class ok_function_vector(Structure):
				_fields_ = [('size', c_size_t),
							('objectives', function_p)]
			function_vector_p = POINTER(ok_function_vector)
			lib.function_vector = ok_function_vector
			lib.function_vector_p = function_vector_p


			# Function Vector
			# ------
			## arguments
			lib.function_vector_alloc.argtypes = [function_vector_p, c_size_t]
			lib.function_vector_calloc.argtypes = [function_vector_p, c_size_t]
			lib.function_vector_free.argtypes = [function_vector_p]
			lib.function_vector_memcpy_va.argtypes = [function_vector_p,
				function_p]
			lib.function_vector_memcpy_av.argtypes = [function_p,
				function_vector_p]
			lib.function_vector_mul.argtypes = [function_vector_p, vector_p]
			lib.function_vector_div.argtypes = [function_vector_p, vector_p]
			lib.function_vector_print.argtypes = [function_vector_p]

			## return values
			lib.function_vector_alloc.restype = None
			lib.function_vector_calloc.restype = None
			lib.function_vector_free.restype = None
			lib.function_vector_memcpy_va.restype = None
			lib.function_vector_memcpy_av.restype = None
			lib.function_vector_mul.restype = None
			lib.function_vector_div.restype = None
			lib.function_vector_print.restype = None

			# Prox & Function evaluation
			# --------------------------
			## arguments
			lib.ProxEvalVector.argtypes = [function_vector_p, ok_float,
				vector_p, vector_p]
			lib.FuncEvalVector.argtypes = [function_vector_p, vector_p]

			## return values
			lib.ProxEvalVector.restype = None
			lib.FuncEvalVector.restype = ok_float

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib