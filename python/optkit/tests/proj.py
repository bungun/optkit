import numpy as np
from traceback import format_exc
from numpy.linalg import norm
from optkit.utils.pyutils import println, pretty_print, printvoid, var_assert
from optkit.tests.defs import gen_test_defs

def direct_proj_test(errors, m, n, A=None, 
	normalize=False, PRINT=lambda x : None):

	try:
		from optkit.api import backend, linsys, Matrix, Vector, DirectProjector
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)


		A = Matrix(RAND_ARR(m,n)) if A is None else Matrix(A)
		ProjA = DirectProjector(A, normalize=normalize)
		assert var_assert(ProjA, type=DirectProjector)
		linsys['sync'](A) # (modified by projector normalization)

		x = Vector(RAND_ARR(n))
		y = Vector(RAND_ARR(m))
		x_out = Vector(n)
		y_out = Vector(m)
		assert var_assert(x, y, x_out, y_out, type=Vector)

		PRINT("RANDOM (x,y)")
		PRINT("||x||_2, {} \t ||y||_2: {}".format(norm(x.py), norm(y.py)))
		PRINT("||Ax-y||_2:")
		PRINT(norm(y.py - A.py.dot(x.py)))

		PRINT("NORM A (from projector): ", ProjA.normA)
		ProjA(x, y, x_out, y_out)
		linsys['sync'](x, y, x_out, y_out)

		PRINT("PROJECT:")
		PRINT("||x||_2, {} \t ||y||_2: {}".format(norm(x_out.py), norm(y_out.py)))
		PRINT("||Ax-y||_2:") 
		res = norm(y_out.py - A.py.dot(x_out.py))
		PRINT(res)

		if backend.lowtypes.FLOAT_CAST == np.float64:
			assert res <= 1e-5
		else:
			assert res <= 5e-2	# non-failing threshold for float32 projection
		
		return True
	except:
		errors.append(format_exc())
		return False

def projector_test(errors, m=None, n=None, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		if m is None: m = 1000
		if n is None: n = 3000
		if isinstance(A_in, np.ndarray):
			if len(A_in.shape) != 2:
				A_in = None
			else:
				A_in = A_in.astype(backend.lowtypes.FLOAT_CAST)
				A_in = A_in if m >= n else A_in.T
				(m,n) = A_in.shape


		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid


		PPRINT("BEGIN TESTING", '=')
		PPRINT("DIRECT PROJECTOR")


		# skinny matrix
		PPRINT("SKINNY MATRIX", '.')
		PRINT("m: {}\tn: {}".format(m,n))
		assert direct_proj_test(errors, m, n, A=A_in, PRINT=PRINT)


		# fat matrix
		PPRINT("FAT MATRIX", '.')
		PRINT("m: {}\tn: {}".format(n,m))
		if isinstance(A_in,np.ndarray): A_in=A_in.T
		assert direct_proj_test(errors, n, m, A=A_in, PRINT=PRINT)


		PPRINT("INDIRECT PROJECTOR")
		PRINT("(NOT IMPLEMENTED)")


		PPRINT("END TESTING",'=')
		return True
	except:
		errors.append(format_exc())
		return False

def test_projector(errors, *args,**kwargs):
	print "PROJECTOR TESTING \n\n\n\n"
	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (None,None)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	success = projector_test(errors, m=m, n=n, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)
	
	if success:
		print "...passed"
	else:
		print "...failed"
	return success