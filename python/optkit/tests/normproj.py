import numpy as np
from optkit.utils.pyutils import println,printvoid
from optkit.types import Matrix, Vector
from optkit.kernels import copy, sync
from optkit.projector import DirectProjector
from optkit.equilibration import *
from optkit.tests.proj import direct_proj_test
from optkit.tests.defs import TEST_EPS


def normalize_and_project_test(m=None,n=None,A_in=None,VERBOSE_TEST=True):
	if m is None: m=1000
	if n is None: n=3000
	if isinstance(A_in,np.ndarray):
		if len(A_in.shape)!=2:
			A_in=None
		else:
			A_in = A_in if m>=n else A_in.T
			(m,n)=A_in.shape

	if not isinstance(A_in,np.ndarray):
		A_in = np.random.rand(m,n)

	PRINT=println if VERBOSE_TEST else printvoid


	PRINT("\n=============")
	PRINT("BEGIN TESTING")
	PRINT("=============")

	PRINT("\n\nDIRECT PROJECTOR")
	PRINT("----------------\n\n")

	# fat matrix
	PRINT("\nFAT MATRIX")
	PRINT("----------\n")
	A = Matrix(A_in)
	A_equil = Matrix(np.zeros_like(A.py))
	d = Vector(m)
	e = Vector(n)

	PRINT("m: {}\tn: {}".format(m,n))
	PRINT("\nORIGINAL MATRIX")
	assert direct_proj_test(m,n,A.py,PRINT=PRINT)

	copy(A,A_equil)
	sync(A_equil)
	PRINT("\nORIGINAL MATRIX, normalized")
	assert direct_proj_test(m,n,A_equil.py,normalize=True,PRINT=PRINT)


	PRINT("\nL2-EQUILIBRATED MATRIX")
	dense_l2_equilibration(A,A_equil,d,e)
	sync(A_equil)
	assert direct_proj_test(m,n,A_equil.py,PRINT=PRINT)

	PRINT("\nL2-EQUILIBRATED MATRIX, normalized")
	dense_l2_equilibration(A,A_equil,d,e)
	sync(A_equil)
	assert direct_proj_test(m,n,A_equil.py,normalize=True,PRINT=PRINT)


	PRINT("\nSINKHORN-EQUILIBRATED MATRIX")
	sinkhornknopp_equilibration(A,A_equil,d,e)
	sync(A_equil)
	assert direct_proj_test(m,n,A_equil.py,PRINT=PRINT)	

	PRINT("\nSINKHORN-EQUILIBRATED MATRIX, normalized")
	sync(A_equil)
	sinkhornknopp_equilibration(A,A_equil,d,e)
	assert direct_proj_test(m,n,A_equil.py,normalize=True,PRINT=PRINT)	


	# skinny matrix
	PRINT("\nSKINNY MATRIX")
	PRINT("-------------\n")

	(m,n) = (n,m)


	PRINT("m: {}\tn: {}".format(m,n))
	PRINT("\nORIGINAL MATRIX")
	assert direct_proj_test(m,n,A.py.T,PRINT=PRINT)

	copy(A,A_equil)
	sync(A_equil)
	PRINT("\nORIGINAL MATRIX, normalized")
	assert direct_proj_test(m,n,A_equil.py.T,normalize=True,PRINT=PRINT)


	PRINT("\nL2-EQUILIBRATED MATRIX")
	dense_l2_equilibration(A,A_equil,d,e)
	sync(A_equil)
	assert direct_proj_test(m,n,A_equil.py.T,PRINT=PRINT)

	PRINT("\nL2-EQUILIBRATED MATRIX, normalized")
	dense_l2_equilibration(A,A_equil,d,e)
	sync(A_equil)
	assert direct_proj_test(m,n,A_equil.py.T,normalize=True,PRINT=PRINT)


	PRINT("\nSINKHORN-EQUILIBRATED MATRIX")
	sinkhornknopp_equilibration(A,A_equil,d,e)
	sync(A_equil)
	assert direct_proj_test(m,n,A_equil.py.T,PRINT=PRINT)	

	PRINT("\nSINKHORN-EQUILIBRATED MATRIX, normalized")
	sinkhornknopp_equilibration(A,A_equil,d,e)
	sync(A_equil)
	assert direct_proj_test(m,n,A_equil.py.T,normalize=True,PRINT=PRINT)	


	PRINT("\n\nINDIRECT PROJECTOR")
	PRINT("------------------\n\n")

	PRINT("(NOT IMPLEMENTED)")

	PRINT("\n===========")
	PRINT("END TESTING")
	PRINT("===========")
	return True

def test_normalizedprojector(*args,**kwargs):
	print "NORMALIZED PROJECTOR TESTING\n\n\n\n"

	verbose = '--verbose' in args
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (None,None)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	assert normalize_and_project_test(m=m,n=n,A_in=A,VERBOSE_TEST=verbose)

	print "...passed"
	return True
