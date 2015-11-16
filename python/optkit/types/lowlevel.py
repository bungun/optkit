from ctypes import c_float, c_double, c_int, c_size_t, \
					POINTER, Structure, c_void_p
from optkit.defs import FLOAT_FLAG
from numpy import dtype

ok_float = c_float if FLOAT_FLAG else c_double

# pointers to C types
c_int_p = POINTER(c_int)
ok_float_p = POINTER(ok_float)

# enums
class OKEnums(object):
	CblasRowMajor = c_int(101)
	CblasColMajor = c_int(102)
	CblasNoTrans = c_int(111)
	CblasTrans = c_int(112)
	CblasConjTrans = c_int(113)
	CblasUpper = c_int(121)
	CblasLower = c_int(122)
	CblasNonUnit = c_int(131)
	CblasUnit = c_int(132)
	CblasLeft = c_int(141)
	CblasRight = c_int(142)
	def __init__(self):
		pass

class OKFunctionEnums(object):
	Zero = c_int(0)
	Abs = c_int(1)
	Exp = c_int(2)
	Huber = c_int(3)
	Identity = c_int(4)
	IndBox01 = c_int(5)
	IndEq0 = c_int(6)
	IndGe0 = c_int(7)
	IndLe0 = c_int(8)
	Logistic = c_int(9)
	MaxNeg0 = c_int(10)
	MaxPos0 = c_int(11)
	NegEntr = c_int(12)
	NegLog = c_int(13)
	Recipr = c_int(14)
	Square = c_int(15)

	def __init__(self):
		self.min_enum = 0;
		self.max_enum = 15;
		self.enum_dict = {'Zero':1}

	def safe_enum(self, h):
		if isinstance(h, (int, c_int)):
			h_ = h if isinstance(h, int) else h.value

			if h_ < self.min_enum or h_ > self.max_enum:
					print ("value out of range, "
							"(valid = {} to {}\n"
							"Setting `h` = Zero".format(
							self.min_enum, self.max_enum))
					return c_int(0)
			else:
				return c_int(h_)
		elif isinstance(h, str):
			if not h in self.enum_dict:
				print ("invalid key. valid keys:\n{}\n"
					   "Setting `h` = Zero)".format(
					   self.enum_dict.keys()))
				return c_int(0)			
			else:
				return c_int(self.enum_dict[h])
		else:
			print ("optkit.types.Function, field \'h\'"
				   "can be initialized with arguments of type:\n"
				   "`int`, `c_int`, or `str`\n"
				   "Setting `h` = Zero")
			return c_int(0)

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
				('rowmajor',c_int)]

# matrix pointer
matrix_p = POINTER(ok_matrix)

# function struct
class ok_function(Structure):
	_fields_ = [('h', c_int_p),
				('a', ok_float_p),
				('b', ok_float_p),
				('c', ok_float_p),
				('d', ok_float_p),
				('e', ok_float_p)]

# function pointer
function_p = POINTER(ok_function)

# function datatype
function_dt = dtype([('h', c_int),('a', ok_float),('b', ok_float),
					 ('c', ok_float), ('d', ok_float), ('e', ok_float)])


# function vector struct
class ok_function_vector(Structure):
	_fields_ = [('size', c_size_t),
				('objectives', c_void_p)]

# function vector pointer
function_vector_p = POINTER(ok_function_vector)



