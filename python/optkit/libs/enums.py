from ctypes import c_uint

class OKEnums(object):
	# C BLAS
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

	# Optkit Transformatiosn
	OkTransformScale = c_uint(0).value
	OkTransformAdd = c_uint(1).value
	OkTransformIncrement = c_uint(2).value
	OkTransformDecrement = c_uint(3).value

	# Optkit Operators
	NULL = 0
	IDENTITY = 101
	DENSE = 201
	SPARSE_CSR = 301
	SPARSE_CSC = 302
	SPARSE_COO = 303
	DIAGONAL = 401

	# Optkit Projectors
	DENSE_DIRECT = 101
	SPARSE_DIRECT = 102
	INDIRECT = 103

	# Errors
	OPTKIT_SUCCESS = 0
	OPTKIT_ERROR = 1
	OPTKIT_ERROR_CUDA = 2
	OPTKIT_ERROR_CUBLAS = 3
	OPTKIT_ERROR_CUSPARSE = 4
	OPTKIT_ERROR_DOMAIN = 10
	OPTKIT_ERROR_DIVIDE_BY_ZERO = 11
	OPTKIT_ERROR_LAYOUT_MISMATCH = 100
	OPTKIT_ERROR_DIMENSION_MISMATCH = 101
	OPTKIT_ERROR_OUT_OF_BOUNDS = 102
	OPTKIT_ERROR_OVERWRITE = 1000
	OPTKIT_ERROR_UNALLOCATED = 1001

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