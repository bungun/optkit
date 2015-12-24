from ctypes import CDLL, c_void_p, c_uint
from os import uname
from optkit.api import backend


from optkit.api import Matrix, Vector
from optkit.types import ok_enums
from optkit.tests.defs import HLINE, TEST_EPS, rand_arr
from optkit.utils.pyutils import println, pretty_print, printvoid, \
	var_assert, array_compare
from sys import argv
from numpy import ndarray, copy as np_copy
from os import path

ok_float_p = backend.lowtypes.ok_float_p
vector_p = backend.lowtypes.vector_p
matrix_p = backend.lowtypes.matrix_p
ndarray_pointer = backend.lowtypes.ndarray_pointer
hdl = backend.dense_blas_handle

DEVICE = backend.device
PRECISION = backend.precision
LAYOUT = backend.layout + '_' if backend.layout != '' else ''
EXT = 'dylib' if uname()[0] == 'Darwin' else 'so'

def main(m = 10, n = 5, A_in=None, VERBOSE_TEST=True):
	PRINT = println if VERBOSE_TEST else printvoid
	HOME = path.dirname(path.abspath(__file__))
	HOME = path.join(HOME, 'build')

	# ------------------------------------------------------------ #
	# ---------------------- libequil prep ----------------------- #
	# ------------------------------------------------------------ #

	libpath = path.join(path.join(HOME, '..', 'build'),
	 'libequil_{}{}{}.{}'.format(LAYOUT, DEVICE, PRECISION, EXT))
	lib = CDLL(libpath)


	lib.sinkhorn_knopp.argtypes = [c_void_p, ok_float_p, matrix_p, 
										vector_p, vector_p, c_uint]
	lib.dense_l2.argtypes = [c_void_p, ok_float_p, matrix_p, 
										vector_p, vector_p, c_uint]

	lib.sinkhorn_knopp.restype = None 
	lib.dense_l2.restype = None



	# ------------------------------------------------------------ #
	# ------------------------- setup ---------------------------- #
	# ------------------------------------------------------------ #
	PRINT("\n")

	if isinstance(A_in, ndarray):
		A = Matrix(A_in)
		var_assert(A)
		(m, n) = A.shape
		PRINT("(using provided matrix)")
	else:
		A = Matrix(rand_arr(m,n))
		var_assert(A)
		PRINT("(using random matrix)")

	pretty_print("{} MATRIX".format("SKINNY" if m >= n else "FAT"), '=')
	print "(m = {}, n = {})".format(m, n)

	# store original
	A_orig = np_copy(A.py)

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
	pretty_print("DENSE L2 EQULIBRATION:")
	lib.dense_l2(hdl, ndarray_pointer(A_orig), A.c, d.c, e.c, order)

	x_rand = rand_arr(n)

	Ax = A.py.dot(x_rand)
	DAEx = d.py * A_orig.dot(e.py * x_rand)

	PRINT("(D * A_equil * E)x  - Ax:")
	PRINT(DAEx - Ax)
	assert array_compare(DAEx, Ax, eps=TEST_EPS)
	



	A2 = Matrix(A_orig)
	assert var_assert(A2)
	# ------------------------------------------------------------ #
	# --------------- SINKHORN KNOPP EQUILIBRATION --------------- #
	# ------------------------------------------------------------ #
	pretty_print("SINKHORN-KNOPP EQUILIBRATION:")
	lib.sinkhorn_knopp(hdl, ndarray_pointer(A_orig), 
		A2.c, d.c, e.c, order)

	Ax_2 = A2.py.dot(x_rand)
	DAEx_2 = d.py * A_orig.dot(e.py * x_rand)

	PRINT("(D * A_equil * E)x  - Ax:")
	PRINT(DAEx - Ax)
	assert array_compare(DAEx, Ax, eps=TEST_EPS)

	PRINT("\n")



if __name__ == '__main__':
	m, n = (10, 5)
	if '--size' in argv:
		pos = argv.index('--size')
		if len(argv) > pos + 2:
			(m, n) = (int(argv[pos+1]),int(argv[pos+2]))
	verbose = '--verbose' in argv
	main(m, n, A_in=None, VERBOSE_TEST=verbose)
	main(n, m, A_in=None, VERBOSE_TEST=verbose)


