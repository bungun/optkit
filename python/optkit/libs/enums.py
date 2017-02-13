from optkit.compat import *

import ctypes as ct

class OKEnums(object):
	# C BLAS
	CblasRowMajor = ct.c_uint(101).value
	CblasColMajor = ct.c_uint(102).value
	CblasNoTrans = ct.c_uint(111).value
	CblasTrans = ct.c_uint(112).value
	CblasConjTrans = ct.c_uint(113).value
	CblasUpper = ct.c_uint(121).value
	CblasLower = ct.c_uint(122).value
	CblasNonUnit = ct.c_uint(131).value
	CblasUnit = ct.c_uint(132).value
	CblasLeft = ct.c_uint(141).value
	CblasRight = ct.c_uint(142).value

	# Optkit Transformatiosn
	OkTransformScale = ct.c_uint(0).value
	OkTransformAdd = ct.c_uint(1).value
	OkTransformIncrement = ct.c_uint(2).value
	OkTransformDecrement = ct.c_uint(3).value

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
	Zero = ct.c_uint(0).value
	Abs = ct.c_uint(1).value
	Exp = ct.c_uint(2).value
	Huber = ct.c_uint(3).value
	Identity = ct.c_uint(4).value
	IndBox01 = ct.c_uint(5).value
	IndEq0 = ct.c_uint(6).value
	IndGe0 = ct.c_uint(7).value
	IndLe0 = ct.c_uint(8).value
	Logistic = ct.c_uint(9).value
	MaxNeg0 = ct.c_uint(10).value
	MaxPos0 = ct.c_uint(11).value
	NegEntr = ct.c_uint(12).value
	NegLog = ct.c_uint(13).value
	Recipr = ct.c_uint(14).value
	Square = ct.c_uint(15).value
	Berhu = ct.c_uint(16).value
	# AffQuad = ct.c_uint(15).value
	__str2fun = {
			'Zero': 0, 'Abs': 1, 'Exp': 2, 'Huber': 3,
			'Identity': 4, 'IndBox01': 5, 'IndEq0': 6, 'IndGe0': 7,
			'IndLe0': 8, 'Logistic': 9, 'MaxNeg0': 10, 'MaxPos0': 11,
			'NegEntr': 12, 'NegLog': 13, 'Recipr': 14, 'Square': 15,
			'Berhu': 16, #'AffQuad': 17,
	}
	min_enum = 0
	max_enum = max(__str2fun.values())

	@property
	def dict(self):
		return self.__str2fun

	def validate(self, h):
		if isinstance(h, int):
			if h < self.min_enum or h > self.max_enum:
				raise IndexError(
						'value out of range: {}, (valid = {} to {}\n'
						''.format(h, self.min_enum, self.max_enum))
			else:
				return ct.c_uint(h).value
		elif isinstance(h, str):
			if not h in self.__str2fun:
				raise KeyError(
						'invalid key: {}. valid keys:\n{}\n'
						''.format(h, self.__str2fun.keys()))
				return ct.c_uint(0).value
			else:
				return ct.c_uint(self.dict[h]).value
		else:
			raise TypeError(
					'optkit.types.Function, field `h` can be '
					'initialized with arguments of type:\n `int`, '
					'`c_int`, or `str`\n')

	def validate_c(self, c):
		if c <= 0:
			raise ValueError(
					'Function parameter `c` must be nonnegative '
					'for function to be convex:\n '
					'\tf(x) =def= c * h(ax - b) + dx + ex^2'
					'\n, with h convex.'.format(name))
		else:
			return c

	def validate_e(self, e):
		if e < 0:
			raise ValueError(
					'Function parameter `e` must be nonnegative '
					'for function to be convex:\n '
					'\tf(x) =def= c * h(ax - b) + dx + ex^2'
					'\n, with h convex.'.format(name))
		else:
			return e

	def validate_s(self, s):
		if s <= 0:
			raise ValueError(
					'Asymmetry parameter `s` must be strictly '
					'positive for function to be convex:\n '
					'\t s =def= 1 * I(ax < b) + scalar * I(ax > b)\n'
					'and\n'
					'\tf(x) =def= c * s * h(ax - b) + dx + ex^2'
					'\n, with h convex.')
		else:
			return s
