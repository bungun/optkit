from traceback import format_exc
import numpy as np
from optkit.utils.pyutils import println, pretty_print, printvoid
from optkit.tests.proj import direct_proj_test
from optkit.tests.defs import gen_test_defs


def normalize_and_project_test(errors, m=None, n=None, A_in=None, 
	VERBOSE_TEST=True, gpu=False, floatbits=64):
	
	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import Vector, Matrix, linsys, equil
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

		if m is None: m=1000
		if n is None: n=3000
		if isinstance(A_in, np.ndarray):
			if len(A_in.shape) != 2:
				A_in = None
			else:
				A_in = A_in if m >= n else A_in.T
				(m,n) = A_in.shape
				A_in = A_in.astype(backend.lowtypes.FLOAT_CAST)

		if not isinstance(A_in, np.ndarray):
			A_in = RAND_ARR(m, n)

		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid


		PPRINT("BEGIN TESTING",'+')

		PPRINT("DIRECT PROJECTOR")

		# fat matrix
		PPRINT("FAT MATRIX",'.')
		A = Matrix(A_in)
		A_equil = Matrix(np.zeros_like(A.py))
		d = Vector(m)
		e = Vector(n)

		PRINT("m: {}\tn: {}".format(m,n))
		PRINT("\nORIGINAL MATRIX")
		assert direct_proj_test(errors, m, n, A.py, PRINT=PRINT)

		linsys['copy'](A,A_equil)
		linsys['sync'](A_equil)
		PRINT("\nORIGINAL MATRIX, normalized")
		assert direct_proj_test(errors, m, n, 
			A_equil.py, normalize=True, PRINT=PRINT)


		PRINT("\nL2-EQUILIBRATED MATRIX")
		equil['dense_l2'](A,A_equil,d,e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py, PRINT=PRINT)

		PRINT("\nL2-EQUILIBRATED MATRIX, normalized")
		equil['dense_l2'](A,A_equil,d,e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py, 
			normalize=True, PRINT=PRINT)


		PRINT("\nSINKHORN-EQUILIBRATED MATRIX")
		equil['sinkhornknopp'](A, A_equil, d, e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py, PRINT=PRINT)	

		PRINT("\nSINKHORN-EQUILIBRATED MATRIX, normalized")
		equil['sinkhornknopp'](A,A_equil,d,e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py, 
			normalize=True, PRINT=PRINT)	


		# skinny matrix
		PPRINT("SKINNY MATRIX",'.')

		(m,n) = (n,m)


		PRINT("m: {}\tn: {}".format(m,n))
		PRINT("\nORIGINAL MATRIX")
		assert direct_proj_test(errors, m, n, A.py.T, PRINT=PRINT)

		linsys['copy'](A,A_equil)
		linsys['sync'](A_equil)
		PRINT("\nORIGINAL MATRIX, normalized")
		assert direct_proj_test(errors, m, n, A_equil.py.T, 
			normalize=True, PRINT=PRINT)


		PRINT("\nL2-EQUILIBRATED MATRIX")
		equil['dense_l2'](A,A_equil,d,e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py.T, PRINT=PRINT)

		PRINT("\nL2-EQUILIBRATED MATRIX, normalized")
		equil['dense_l2'](A,A_equil,d,e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py.T, 
			normalize=True, PRINT=PRINT)


		PRINT("\nSINKHORN-EQUILIBRATED MATRIX")
		equil['sinkhornknopp'](A,A_equil,d,e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py.T, PRINT=PRINT)	

		PRINT("\nSINKHORN-EQUILIBRATED MATRIX, normalized")
		equil['sinkhornknopp'](A,A_equil,d,e)
		linsys['sync'](A_equil)
		assert direct_proj_test(errors, m, n, A_equil.py.T,
			normalize=True, PRINT=PRINT)	


		PPRINT("INDIRECT PROJECTOR")

		PRINT("(NOT IMPLEMENTED)")

		PPRINT("END TESTING",'=')
		return True
		
	except:
		errors.append(format_exc())
		return False

def test_normalizedprojector(errors, *args,**kwargs):
	print "NORMALIZED PROJECTOR TESTING\n\n\n\n"

	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64	
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (None,None)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	success = normalize_and_project_test(errors, m=m,n=n,A_in=A,
		VERBOSE_TEST=verbose, gpu='gpu' in args, floatbits=floatbits)

	if success:
		print "...passed"
	else:
		print "...failed"
	return success