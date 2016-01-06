from optkit.api import *
from optkit.api import backend
from optkit.utils.pyutils import println, printvoid, var_assert
from optkit.tests.defs import TEST_EPS, MAT_ORDER, rand_arr
import numpy as np



def direct_proj_test(m,n,A=None,normalize=False,PRINT=lambda x : None):
	A = Matrix(rand_arr(m,n)) if A is None else Matrix(A)
	ProjA = DirectProjectorPy(A, normalize=normalize)
	assert var_assert(ProjA, type=DirectProjectorPy)
	sync(A) #(since potentially modified by projector normalization)

	x = Vector(rand_arr(n))
	y = Vector(rand_arr(m))
	x_out = Vector(n)
	y_out = Vector(m)
	assert var_assert(x,y,x_out,y_out,type=Vector)

	PRINT("RANDOM (x,y)")
	PRINT("||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x.py), np.linalg.norm(y.py)))
	PRINT("||Ax-y||_2:")
	PRINT(np.linalg.norm(y.py-A.py.dot(x.py)))

	PRINT("NORM A (from projector): ", ProjA.normA)
	ProjA(x, y, x_out, y_out)
	sync(x, y, x_out, y_out)

	PRINT("PROJECT:")
	PRINT("||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x_out.py), np.linalg.norm(y_out.py)))
	PRINT("||Ax-y||_2:")
	res = np.linalg.norm(y_out.py-A.py.dot(x_out.py))
	PRINT(res)


	if backend.lowtypes.FLOAT_CAST == np.float64:
		assert res <= TEST_EPS
	else:
		# non-failing threshold for 32-bit float projection
		assert res <= 5e-2		
	return True

def projector_test(m=None,n=None,A_in=None,VERBOSE_TEST=True):
	if m is None: m=1000
	if n is None: n=3000
	if isinstance(A_in,np.ndarray):
		if len(A_in.shape)!=2:
			A_in=None
		else:
			A_in = A_in if m>=n else A_in.T
			(m,n)=A_in.shape


	PRINT=println if VERBOSE_TEST else printvoid

	PRINT("\n=============")
	PRINT("BEGIN TESTING")
	PRINT("=============")

	PRINT("\n\nDIRECT PROJECTOR")
	PRINT("----------------\n\n")


	# skinny matrix
	PRINT("\nSKINNY MATRIX")
	PRINT("-------------\n")
	PRINT("m: {}\tn: {}".format(m,n))
	assert direct_proj_test(m,n,A=A_in,PRINT=PRINT)


	# fat matrix
	PRINT("\nFAT MATRIX")
	PRINT("----------\n")
	PRINT("m: {}\tn: {}".format(n,m))
	if isinstance(A_in,np.ndarray): A_in=A_in.T
	assert direct_proj_test(n,m,A=A_in,PRINT=PRINT)


	PRINT("\n\nINDIRECT PROJECTOR")
	PRINT("------------------\n\n")

	PRINT("(NOT IMPLEMENTED)")


	PRINT("\n===========")
	PRINT("END TESTING")
	PRINT("===========")
	return True

def test_projector(*args,**kwargs):
	print "PROJECTOR TESTING \n\n\n\n"
	verbose = '--verbose' in args
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (None,None)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	assert projector_test(m=m,n=n,A_in=A,VERBOSE_TEST=verbose)
	print "...passed"
	return True