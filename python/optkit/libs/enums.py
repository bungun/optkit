from optkit.compat import *

import ctypes as ct

class OKEnums:
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

    MATRIX_ORDERS = (CblasRowMajor, CblasColMajor)

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

    # Optkit POGS Implementations
    OkPogsDense = 1001
    OkPogsSparse = 1002
    OkPogsAbstract = 2001

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

    def get_layout(self, array):
        # SCIPY SPARSE CSR/CSC
        if hasattr(array, 'format'):
            if array.format == 'csr':
                return self.CblasRowMajor
            if array.format == 'csc':
                return self.CblasColMajor
        # NUMPY NDARRAY
        if hasattr(array, 'flags'):
            flags = array.flags
            try:
                if getattr(flags, 'c_contiguous'):
                    return self.CblasRowMajor
                else:
                    return self.CblasColMajor
            except:
                pass
        raise ValueError('array of unknown layout')

class OKFunctionEnums:
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
    AsymmHuber = ct.c_uint(17).value
    AsymmSquare = ct.c_uint(18).value
    AsymmBerhu = ct.c_uint(19).value
    AbsQuad = ct.c_uint(20).value
    AbsExp = ct.c_uint(21).value
    __str2fun = {
            'Zero': 0, 'Abs': 1, 'Exp': 2, 'Huber': 3,
            'Identity': 4, 'IndBox01': 5, 'IndEq0': 6, 'IndGe0': 7,
            'IndLe0': 8, 'Logistic': 9, 'MaxNeg0': 10, 'MaxPos0': 11,
            'NegEntr': 12, 'NegLog': 13, 'Recipr': 14, 'Square': 15,
            'Berhu': 16, 'AsymmHuber': 17, 'AsymmSquare': 18,
            'AsymmBerhu': 19,  'AbsQuad': 20, 'AbsExp': 21,
    }
    min_enum = 0
    max_enum = max(__str2fun.values())

    @property
    def dict(self):
        return self.__str2fun

    def validator(self, key):
        if key == 'h':
            return self.validate_func
        elif key == 'c':
            return self.validate_c
        elif key == 'e':
            return self.validate_e
        elif key == 's':
            return self.validate_s
        else:
            return lambda val: val

    def validate_func(self, h):
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
        if c < 0:
            raise ValueError(
                    'Function parameter `c` must be nonnegative '
                    'for function to be convex:\n '
                    '\tf(x) =def= c * h(ax - b) + dx + ex^2,'
                    '\nwith h convex.')
        else:
            return c

    def validate_e(self, e):
        if e < 0:
            raise ValueError(
                    'Function parameter `e` must be nonnegative '
                    'for function to be convex:\n '
                    '\tf(x) =def= c * h(ax - b) + dx + ex^2,'
                    '\nwith h convex.')
        else:
            return e

    def validate_s(self, s):
        if s <= 0:
            raise ValueError(
                    'Asymmetry parameter `s` must be strictly '
                    'positive for function to be convex:\n '
                    '\t s =def= 1 * I(ax < b) + scalar * I(ax > b)\n'
                    'and\n'
                    '\tf(x) =def= c * s * h(ax - b) + dx + ex^2,'
                    '\nwith h convex.')
        else:
            return s

class OKOperatorEnums(object):
    OkOperatorNull = ct.c_uint(0).value,
    OkOperatorIdentity = ct.c_uint(101).value,
    OkOperatorNeg = ct.c_uint(102).value,
    OkOperatorAdd = ct.c_uint(103).value,
    OkOperatorCat = ct.c_uint(104).value,
    OkOperatorSplit = ct.c_uint(105).value,
    OkOperatorDense = ct.c_uint(201).value,
    OkOperatorSparseCSR = ct.c_uint(301).value,
    OkOperatorSparseCSC = ct.c_uint(302).value,
    OkOperatorSparseCOO = ct.c_uint(303).value,
    OkOperatorDiagonal = ct.c_uint(401).value,
    OkOperatorBanded = ct.c_uint(402).value,
    OkOperatorTriangular = ct.c_uint(403).value,
    OkOperatorKronecker = ct.c_uint(404).value,
    OkOperatorToeplitz = ct.c_uint(405).value,
    OkOperatorCirculant = ct.c_uint(406).value,
    OkOperatorConvolution = ct.c_uint(501).value,
    OkOperatorCircularConvolution = ct.c_uint(502).value,
    OkOperatorFourier = ct.c_uint(503).value,
    OkOperatorDifference = ct.c_uint(504).value,
    OkOperatorUpsampling = ct.c_uint(505).value,
    OkOperatorDownsampling = ct.c_uint(506).value,
    OkOperatorBlockDifference = ct.c_uint(507).value,
    OkOperatorDirectProjection = ct.c_uint(901).value,
    OkOperatorIndirectProjection = ct.c_uint(902).value,
    OkOperatorOther = ct.c_uint(1000).value
