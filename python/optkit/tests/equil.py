from traceback import format_exc
from operator import and_
import numpy as np
from optkit.tests.defs import gen_test_defs
from optkit.utils.pyutils import println, pretty_print, printvoid, var_assert

def dense_equil_test(errors, equil_method, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import Vector, Matrix, linsys, equil
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

		PRINT = println if VERBOSE_TEST else printvoid
		PNORM = 1 if equil_method == 'sinkhornknopp' else 2


		PRINT("SKINNY MATRIX")

		if isinstance(A_in, np.ndarray):
			if A_in.shape[1] > A_in.shape[0]: A_in = A_in.T
			A = Matrix(A_in.astype(backend.lowtypes.FLOAT_CAST))
			(m1,n1) = A.shape
		else:
			(m1,n1) = (50, 30)
			A = Matrix(RAND_ARR(m1,n1))
		A_out = Matrix(np.zeros_like(A.py))
		A_orig= np.copy(A.py)


		PRINT("m: {}\tn: {}".format(m1,n1))

		d = Vector(m1)
		e = Vector(n1)

		assert var_assert(A,A_out,type=Matrix)
		assert var_assert(d,e,type=Vector)

		PRINT("\nBEFORE")

		PRINT("row norms A:")
		for i in xrange(m1):
			PRINT("{}: {}".format(i, np.linalg.norm(A.py[i,:])))
		PRINT("column norms A:")
		for j in xrange(n1):
			PRINT("{}: {}".format(j, np.linalg.norm(A.py[:,j])))

		PRINT("d: ", d.py)
		PRINT("e: ", e.py)
		equil[equil_method](A, A_out, d, e)
		linsys['sync'](A, A_out, d, e)


		PRINT("\nAFTER")


		PRINT("row {}-norms A:".format(PNORM))
		for i in xrange(m1):
			PRINT("{}: {}".format(i, np.linalg.norm(A_out.py[i,:],PNORM)))
		PRINT("column {}-norms A:".format(PNORM))
		for j in xrange(n1):
			PRINT("{}: {}".format(j, np.linalg.norm(A_out.py[:,j],PNORM)))


		PRINT("d: ", d.py)
		PRINT("e: ", e.py)

		# equilibration should only change output matrix
		assert reduce(and_,map(lambda x: x==0, [dij for di \
											in A_orig-A.py for dij in di]))


		# verify A_out = DAE
		xrand = RAND_ARR(n1)
		assert all((A_out.py.dot(xrand)-d.py*A_orig.dot(e.py*xrand)) <= TEST_EPS)


		PRINT("FAT MATRIX")


		if A_in is not None:
			B = Matrix(A_in.T)
			(m2,n2) = B.shape
		else:
			(m2,n2) = (30, 50)
			B = Matrix(RAND_ARR(m2,n2))
		B_out = Matrix(np.zeros_like(B.py))
		B_orig = np.copy(B.py)

		PRINT("m: {}\tn: {}".format(m2,n2))


		d = Vector(m2)
		e = Vector(n2)

		assert var_assert(B,B_out,type=Matrix)
		assert var_assert(d,e,type=Vector)


		PRINT("\nBEFORE")

		PRINT("row {}-norms A:".format(PNORM))
		for i in xrange(m2):
			PRINT("{}: {}".format(i, np.linalg.norm(B.py[i,:],PNORM)))
		PRINT("column {}-norms A:".format(PNORM))
		for j in xrange(n2):
			PRINT("{}: {}".format(j, np.linalg.norm(B.py[:,j],PNORM)))

		PRINT("d: ", d.py)
		PRINT("e: ", e.py)
		equil[equil_method](B, B_out, d, e)
		linsys['sync'](B, B_out, d, e)

		PRINT("\nAFTER")

		PRINT("row norms A:")
		for i in xrange(m2):
			PRINT("{}: {}".format(i, np.linalg.norm(B_out.py[i,:])))
		PRINT("column norms A:")
		for j in xrange(n2):
			PRINT("{}: {}".format(j, np.linalg.norm(B_out.py[:,j])))

		PRINT("d: ", d.py)
		PRINT("e: ", e.py)

		# equilibration should only change output matrix
		assert reduce(and_,map(lambda x: x==0, [dij for \
											di in B_orig-B.py for dij in di]))


		# verify B_out = DBE
		xrand = RAND_ARR(n2)
		assert all((B_out.py.dot(xrand)-d.py*B_orig.dot(e.py*xrand)) <= TEST_EPS)

		return True

	except:
		errors.append(format_exc())
		return False


def test_equil(errors, *args,**kwargs):
	print "EQUILIBRATION TESTING\n\n\n\n\n"
	
	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64
	A = np.load(kwargs['file']) if 'file' in kwargs else None

	pretty_print("DENSE EQUIL", '=')
	success = dense_equil_test(errors, 'dense_l2', A_in=A, 
		VERBOSE_TEST=verbose, gpu='gpu' in args, floatbits=floatbits)

	pretty_print("SINKHORN KNOPP", '=')
	success &= dense_equil_test(errors, 'sinkhornknopp', A_in=A, 
		VERBOSE_TEST=verbose, gpu='gpu' in args, floatbits=floatbits)

	if success:
		print "...passed"
	else:
		print "...failed"
	return success