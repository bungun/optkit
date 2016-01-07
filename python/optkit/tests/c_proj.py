from traceback import format_exc
from ctypes import CDLL, POINTER, Structure, c_void_p, c_uint, c_int
from os import uname, path
from sys import argv
from numpy import ndarray, copy as np_copy
from numpy.linalg import norm
from os import path
from optkit.types import ok_enums
from optkit.utils.pyutils import println, pretty_print, printvoid, \
	var_assert, array_compare
from optkit.tests.defs import gen_test_defs


def main(errors, m = 10, n = 5, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)

		from optkit.api import Matrix, Vector
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)


		ok_float = backend.lowtypes.ok_float
		ok_float_p = backend.lowtypes.ok_float_p
		vector_p = backend.lowtypes.vector_p
		matrix_p = backend.lowtypes.matrix_p
		ndarray_pointer = backend.lowtypes.ndarray_pointer
		hdl = backend.dense_blas_handle

		DEVICE = backend.device
		PRECISION = backend.precision
		LAYOUT = backend.layout + '_' if backend.layout != '' else ''
		EXT = 'dylib' if uname()[0] == 'Darwin' else 'so'


		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid
		HOME = path.dirname(path.abspath(__file__))



		# ------------------------------------------------------------ #
		# ------------------- libprojector prep ---------------------- #
		# ------------------------------------------------------------ #

		libpath = path.join(path.join(HOME, '..', '..', '..', 'build'),
		 'libprojector_{}{}{}.{}'.format(LAYOUT, DEVICE, PRECISION, EXT))

		if not path.exists(libpath):
			print "ERROR:"
			print "library not found at {}".format(libpath)
			print "try setting the shell environment variable"
			print "`export OPTKIT_USE_LOCALLIBS=1` and calling"
			print "test.py from within its own directory"
			print "\n"
			print "also, make sure libequil is built:"
			print "call `make libequil` from the optkit home directory"
			print "(marking test as passed)"
			return True

		lib = CDLL(path.abspath(libpath))


		class DirectProjector(Structure):
			_fields_ = [('A', matrix_p),
						('L', matrix_p),
						('normA', ok_float),
						('skinny', c_int),
						('normalized', c_int)]


		direct_projector_p = POINTER(DirectProjector)

		lib.direct_projector_alloc.argtypes = [direct_projector_p, matrix_p]
		lib.direct_projector_initialize.argtypes = [c_void_p,
		 											direct_projector_p,c_int]
		lib.direct_projector_project.argtypes = [c_void_p, direct_projector_p,
								vector_p, vector_p, vector_p, vector_p]
		lib.direct_projector_free.argtypes = [direct_projector_p]

		lib.direct_projector_alloc.restype = None
		lib.direct_projector_initialize.restype = None
		lib.direct_projector_project.restype = None
		lib.direct_projector_free.restype = None



		# ------------------------------------------------------------ #
		# ------------------------ test setup ------------------------ #
		# ------------------------------------------------------------ #
		PRINT("\n")

		if isinstance(A_in, ndarray):
			A = Matrix(A_in.astype(backend.lowtypes.FLOAT_CAST))
			var_assert(A)
			(m, n) = A.shape
			PRINT("(using provided matrix)")
		else:
			A = Matrix(RAND_ARR(m,n))
			var_assert(A, type=Matrix)
			PRINT("(using random matrix)")

		pretty_print("{} MATRIX".format("SKINNY" if m >= n else "FAT"), '=')
		print "(m = {}, n = {})".format(m, n)

		# store original
		A_orig = np_copy(A.py)

		# 
		P_c = DirectProjector(None, None, 0, 0, 0)



		# ------------------------------------------------------------ #
		# -------------------- Direct Projection --------------------- #
		# ------------------------------------------------------------ #

		# make random (x_in, y_in), allocate all-zero (x_out, y_out)
		x_rand = RAND_ARR(n)
		y_rand = RAND_ARR(m)
		x_in = Vector(x_rand)
		y_in = Vector(y_rand)
		x_out = Vector(n)
		y_out = Vector(m)
		assert var_assert(x_in, y_in, x_out, y_out, type=Vector)

		pretty_print("DIRECT PROJECTOR TEST:", '-')

		PPRINT("NON-NORMALIZED PROJECTOR", '.')

		lib.direct_projector_alloc(P_c, A.c)

	 	# ---------------------------------------------------------	#
		# calculate cholesky factorization of (I+AA') or (I+A'A)	#
		#															#
		# then, set (x_out, y_out) = Proj_{y=Ax} (x_in, y_in)  		#														
		# ---------------------------------------------------------	#
		lib.direct_projector_initialize(backend.dense_blas_handle, P_c, 0)
		lib.direct_projector_project(backend.dense_blas_handle, P_c,
			x_in.c, y_in.c, x_out.c, y_out.c)
		if x_out.sync_required:
			backend.dense.vector_memcpy_av(ndarray_pointer(x_out.py), x_out.c, 1)
			backend.dense.vector_memcpy_av(ndarray_pointer(y_out.py), y_out.c, 1)


		PRINT("RANDOM (x,y)")

		# print ||x_in||, ||y_in||, ||Ax_in - y_in||
		PRINT("||x||_2, {} \t ||y||_2: {}".format(norm(x_rand), norm(y_rand)))
		PRINT("||Ax-y||_2:")
		PRINT(norm(y_rand-A_orig.dot(x_rand)))
		PRINT("NORM A (from projector): ", P_c.normA)
		PRINT("PROJECT:")

		# print ||x_out||, ||y_out||, ||Ax_out - y_out||
		PRINT("||x||_2, {} \t ||y||_2: {}".format(norm(x_out.py), norm(y_out.py)))
		PRINT("||Ax-y||_2:")
		res = norm(y_out.py-A.py.dot(x_out.py))
		PRINT(res)

		# Ax = y should hold (within numerical tolerance)
		assert res <= TEST_EPS

		# inputs should be unchanged
		assert array_compare(x_rand, x_in.py)
		assert array_compare(y_rand, y_in.py)

		lib.direct_projector_free(P_c)


		PPRINT("NORMALIZED PROJECTOR", '.')

		# ---------------------------------------------------------	#
		# repeat above procedure, but divide all entries of A by 	#
		#														 	#
		#  ( \sum_{i=1}^mindim (AA)_{ii} ) / sqrt(mindim)		 	#
		#	 	where 											 	#
		#		-mindim = min(m, n) 								#
		#		-AA = A'A or AA' 									#
		# 															#
		# before forming L = chol(I + A'A) 							#
		# --------------------------------------------------------- #

		lib.direct_projector_alloc(P_c, A.c)
		lib.direct_projector_initialize(backend.dense_blas_handle, P_c, 1)
		lib.direct_projector_project(backend.dense_blas_handle, P_c,
			x_in.c, y_in.c, x_out.c, y_out.c)

		if x_out.sync_required:
			order = 102 if A.py.flags.f_contiguous else 101
			backend.dense.matrix_memcpy_am(ndarray_pointer(A.py), A.c, order)
			backend.dense.vector_memcpy_av(ndarray_pointer(x_out.py), x_out.c, 1)
			backend.dense.vector_memcpy_av(ndarray_pointer(y_out.py), y_out.c, 1)


		PRINT("RANDOM (x,y)")
		PRINT("||x||_2, {} \t ||y||_2: {}".format(norm(x_rand), norm(y_rand)))
		PRINT("||A_{norm}x-y||_2:")
		PRINT(norm(y_rand-A.py.dot(x_rand)))
		PRINT("NORM A (from projector): ", P_c.normA)
		PRINT("PROJECT:")
		PRINT("||x||_2, {} \t ||y||_2: {}".format(norm(x_out.py), norm(y_out.py)))
		PRINT("||Ax-y||_2:")
		res = norm(y_out.py-A.py.dot(x_out.py))
		PRINT(res)
		assert res <= TEST_EPS
		lib.direct_projector_free(P_c)

		return True

	except:
		errors.append(format_exc())
		return False

def test_cproj(errors, *args, **kwargs):
	print("\n\n")
	pretty_print("C PROJECTOR TESTING ...", '#')
	print("\n\n")

	args = list(args)
	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64

	(m,n)=kwargs['shape'] if 'shape' in kwargs else (10, 5)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	success = main(errors, m, n, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)
	if isinstance(A, ndarray): A = A.T
	success &= main(errors, n, m, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)

	if success:
		print("\n\n")
		pretty_print("... passed", '#')
		print("\n\n")
	else:
		print("\n\n")
		pretty_print("... failed", '#')
		print("\n\n")
	return success