from ctypes import c_float, c_double, c_int, c_uint, c_size_t, \
					POINTER, Structure, c_void_p
from numpy import int32, float32, float64, ndarray


class LowLevelTypes(object):
	def __init__(self, single_precision=False, 
		long_int=False, order=''):
		self.ok_float = c_float if single_precision else c_double
		self.ok_int = c_int
		self.order = order
 		self.FLOAT_CAST = float32 if single_precision else float64

		# pointers to C types
		self.c_int_p = POINTER(c_int)
		self.ok_float_p = POINTER(self.ok_float)
		self.ok_int_p = POINTER(self.ok_int)

		# low-level optkit types
		# ----------------------

		# vector struct
		class ok_vector(Structure):
			_fields_ = [('size', c_size_t),
						('stride', c_size_t),
						('data', self.ok_float_p)]

		self.vector = ok_vector
		# vector pointer
		self.vector_p = POINTER(self.vector)

		# matrix struct
		class ok_matrix(Structure):
			_fields_ = [('size1', c_size_t),
						('size2', c_size_t),
						('ld', c_size_t),
						('data', self.ok_float_p),
						('order', c_uint)]
		self.matrix = ok_matrix

		# matrix pointer
		self.matrix_p = POINTER(self.matrix)

		# sparse matrix struct
		class ok_sparse_matrix(Structure):
			_fields_ = [('size1', c_size_t),
						('size2', c_size_t),
						('nnz', c_size_t),
						('ptrlen', c_size_t),
						('val', self.ok_float_p),
						('ind', self.ok_int_p),
						('ptr', self.ok_int_p),
						('order', c_uint)]

		self.sparse_matrix = ok_sparse_matrix

		# sparse matrix pointer
		self.sparse_matrix_p = POINTER(self.sparse_matrix)

		# function struct
		class ok_function(Structure):
			_fields_ = [('h', c_uint),
						('a', self.ok_float),
						('b', self.ok_float),
						('c', self.ok_float),
						('d', self.ok_float),
						('e', self.ok_float)]
		self.function = ok_function

		# function pointer
		self.function_p = POINTER(self.function)

		# function vector struct
		class ok_function_vector(Structure):
			_fields_ = [('size', c_size_t),
						('objectives', self.function_p)]
		self.function_vector = ok_function_vector

		# function vector pointer
		self.function_vector_p = POINTER(self.function_vector)


		def ndarray_pointer(x):
			if not isinstance(x, ndarray):
				raise TypeError("input to method `ndarray_pointer "
					  "must be a NumPy array. \n "
					  "Input type: {}".format(type(x)))
				return None
			if x.dtype == self.FLOAT_CAST:
				return x.ctypes.data_as(self.ok_float_p)
			elif x.dtype == int32:
				return x.ctypes.data_as(self.ok_int_p)
			elif x.dtype == self.function:
				return x.ctypes.data_as(self.function_p)
			else:
				raise ValueError("input to method `ndarray_pointer "
					  "must be a NumPy array of type {}, {} or {}. \n "
					  "Input array type: {}".format(self.FLOAT_CAST,
					  	int32, self.function, x.dtype))
				return None

		self.ndarray_pointer = ndarray_pointer
	



# enums
class OKEnums(object):
	CblasRowMajor = c_uint(101).value
	CblasColMajor = c_uint(102).value
	CblasNoTrans = c_uint(111).value
	CblasTrans = c_uint(112).value
	CblasConjTrans = c_uint(113).value
	CblasUpper = c_uint(121).value
	CblasLower = c_uint(122).value
	CblasNonUnit = c_uint(131).value
	CblasUnit = c_uint(132).value
	CblasLeft = c_uint(141).value
	CblasRight = c_uint(142).value
	def __init__(self):
		pass

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

	def __init__(self):
		self.min_enum = 0;
		self.max_enum = 15;
		self.enum_dict = {'Zero':0, 'Abs':1, 'Exp':2, 'Huber':3, 
			'Identity':4, 'IndBox01':5, 'IndEq0':6, 'IndGe0':7,
			'IndLe0':8, 'Logistic':9, 'MaxNeg0':10, 'MaxPos0':11,
			'NegEntr':12, 'NegLog':13, 'Recipr':14, 'Square':15}

	def safe_enum(self, h):
		if isinstance(h, int):
			if h < self.min_enum or h > self.max_enum:
				print ("value out of range: {}, "
							"(valid = {} to {}\n"
							"Setting `h` = Zero".format(h,
							self.min_enum, self.max_enum))
				return c_uint(0).value
			else:
				return c_uint(h).value
		elif isinstance(h, str):
			if not h in self.enum_dict:
				print ("invalid key: {}. valid keys:\n{}\n"
					   "Setting `h` = Zero)".format(h,
					   self.enum_dict.keys()))
				return c_uint(0).value			
			else:
				return c_uint(self.enum_dict[h]).value
		else:
			print ("optkit.types.Function, field \'h\'"
				   "can be initialized with arguments of type:\n"
				   "`int`, `c_int`, or `str`\n"
				   "Setting `h` = Zero")
			return c_uint(0).value

ok_enums = OKEnums()
ok_function_enums = OKFunctionEnums()