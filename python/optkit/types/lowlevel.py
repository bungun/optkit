from ctypes import c_float, c_double, c_int, c_uint, c_size_t, \
					POINTER, Structure, c_void_p
from optkit.defs import FLOAT_FLAG
from numpy import dtype

ok_float = c_float if FLOAT_FLAG else c_double

# pointers to C types
c_int_p = POINTER(c_int)
c_uint_p = POINTER(c_uint)
ok_float_p = POINTER(ok_float)

# enums
class OKEnums(object):
	CblasRowMajor = c_uint(101)
	CblasColMajor = c_uint(102)
	CblasNoTrans = c_uint(111)
	CblasTrans = c_uint(112)
	CblasConjTrans = c_uint(113)
	CblasUpper = c_uint(121)
	CblasLower = c_uint(122)
	CblasNonUnit = c_uint(131)
	CblasUnit = c_uint(132)
	CblasLeft = c_uint(141)
	CblasRight = c_uint(142)
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


# low-level optkit types
# ----------------------

# vector struct
class ok_vector(Structure):
	_fields_ = [('size', c_size_t),('stride', c_size_t),('data', ok_float_p)]

# vector pointer
vector_p = POINTER(ok_vector)

# matrix struct
class ok_matrix(Structure):
	_fields_ = [('size1',c_size_t),
				('size2',c_size_t),
				('tda',c_size_t),
				('data',ok_float_p),
				('rowmajor',c_uint)]

# matrix pointer
matrix_p = POINTER(ok_matrix)

# function struct
class ok_function(Structure):
	_fields_ = [('h', c_uint),
				('a', ok_float),
				('b', ok_float),
				('c', ok_float),
				('d', ok_float),
				('e', ok_float)]

# function pointer
function_p = POINTER(ok_function)

class ok_function_vector(Structure):
	_fields_ = [('size',c_size_t),
				('objectives',function_p)]

# function vector pointer
function_vector_p = POINTER(ok_function_vector)







