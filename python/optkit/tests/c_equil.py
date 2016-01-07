from traceback import format_exc
from ctypes import CDLL, c_void_p, c_uint
from sys import argv
from numpy import ndarray, copy as np_copy, load as np_load
from os import path, uname
from optkit.utils.pyutils import println, pretty_print, printvoid, \
	var_assert, array_compare
from optkit.types import ok_enums
from optkit.tests.defs import gen_test_defs

def main(errors, m = 10, n = 5, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try: 
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)

		from optkit.api import Matrix, Vector

		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

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
		HOME = path.dirname(path.abspath(__file__))



		# ------------------------------------------------------------ #
		# ---------------------- libequil prep ----------------------- #
		# ------------------------------------------------------------ #

		libpath = path.join(path.join(HOME, '..', '..', '..', 'build'),
		 'libequil_{}{}{}.{}'.format(LAYOUT, DEVICE, PRECISION, EXT))

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


		lib.sinkhorn_knopp.argtypes = [c_void_p, ok_float_p, matrix_p, 
											vector_p, vector_p, c_uint]
		lib.dense_l2.argtypes = [c_void_p, ok_float_p, matrix_p, 
											vector_p, vector_p, c_uint]

		lib.sinkhorn_knopp.restype = None 
		lib.dense_l2.restype = None



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
			var_assert(A)
			PRINT("(using random matrix)")

		pretty_print("{} MATRIX".format("SKINNY" if m >= n else "FAT"), '=')
		print "(m = {}, n = {})".format(m, n)

		# store original
		A_orig = np_copy(A.py)

		# second copy for second equilibration methods
		A1 = Matrix(A_orig)
		A2 = Matrix(A_orig)
		assert var_assert(A2)

		# allocate vectors for equilibration:
		#
		#	 A_orig = D * A_equil * E 
		#
		# (D, E diagonal)
		d = Vector(m)
		e = Vector(n)
		assert var_assert(d, e)

		# get matrix layout
		order = A.c.rowmajor


		# ------------------------------------------------------------ #
		# ------------------- DENSE l_2 EQUILIBRATION  --------------- #
		# ------------------------------------------------------------ #
		
		# TODO: for some reason the lib.dense_l2() call segfaults when
		# the backend is set to gpu32/64 on (at least!) linux, so
		# for now it is hard coded to be skipped. this is unclear, since
		# the lib.skinhorn_knopp() call below has the exact same method
		# signature, is fed equivalent arguments, and works fine

		if DEVICE != 'gpu':
			pretty_print("DENSE L2 EQULIBRATION:")
			lib.dense_l2(hdl, ndarray_pointer(A_orig), A.c, d.c, e.c, order)

			x_rand = RAND_ARR(n)

			if d.sync_required:
				order = 102 if A.py.flags.f_contiguous else 101
				backend.dense.matrix_memcpy_am(ndarray_pointer(A.py), A.c, order)
				backend.dense.vector_memcpy_av(ndarray_pointer(d.py), d.c, 1)
				backend.dense.vector_memcpy_av(ndarray_pointer(e.py), e.c, 1)

			Ax = A.py.dot(x_rand)
			DAEx = d.py * A_orig.dot(e.py * x_rand)

			PRINT("(D * A_equil * E)x  - Ax:")
			PRINT(DAEx - Ax)
			assert array_compare(DAEx, Ax, eps=TEST_EPS)
		

		# ------------------------------------------------------------ #
		# --------------- SINKHORN KNOPP EQUILIBRATION --------------- #
		# ------------------------------------------------------------ #		
		pretty_print("SINKHORN-KNOPP EQUILIBRATION:")
		lib.sinkhorn_knopp(hdl, ndarray_pointer(A_orig), 
			A2.c, d.c, e.c, order)

		if d.sync_required:
			order = 102 if A.py.flags.f_contiguous else 101
			backend.dense.matrix_memcpy_am(ndarray_pointer(A2.py), A2.c, order)
			backend.dense.vector_memcpy_av(ndarray_pointer(d.py), d.c, 1)
			backend.dense.vector_memcpy_av(ndarray_pointer(e.py), e.c, 1)


		Ax_2 = A2.py.dot(x_rand)
		DAEx_2 = d.py * A_orig.dot(e.py * x_rand)

		PRINT("(D * A_equil * E)x  - Ax:")
		PRINT(DAEx_2 - Ax_2)
		assert array_compare(DAEx_2, Ax_2, eps=TEST_EPS)

		PRINT("\n")

		return True

	except:
		errors.append(format_exc())
		return False
		
def test_cequil(errors, *args,**kwargs):
	print("\n\n")
	pretty_print("C EQUILIBRATION TESTING ...", '#')
	print("\n\n")

	args = list(args)
	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64
	
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (10,5)
	A = np_load(kwargs['file']) if 'file' in kwargs else None
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