import numpy as np
from optkit.types import Matrix, Vector
from optkit.utils.pyutils import println,printvoid, var_assert
from optkit.equilibration import *
from optkit.tests.defs import TEST_EPS
from operator import and_

def dense_equil_test(equil_method,A_in=None,VERBOSE_TEST=True):
	PRINT = println if VERBOSE_TEST else printvoid
	PNORM = 1 if equil_method==sinkhornknopp_equilibration else 2


	PRINT("SKINNY MATRIX")

	if A_in is not None:
		A = Matrix(A_in)
		(m1,n1) = A.shape
	else:
		(m1,n1) = (50, 30)
		A = Matrix(np.random.rand(m1,n1))
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
		PRINT(i, ": ", np.linalg.norm(A.py[i,:]))
	PRINT("column norms A:")
	for j in xrange(n1):
		PRINT(j, ": ", np.linalg.norm(A.py[:,j]))

	PRINT("d: ", d.py)
	PRINT("e: ", e.py)
	equil_method(A,A_out,d,e)
	PRINT("\AFTER")

	# equilibration should only change output matrix
	assert reduce(and_,map(lambda x: x==0, [dij for di \
										in A_orig-A.py for dij in di]))


	# verify A_out = DAE
	xrand = np.random.rand(n1)
	assert all((A_out.py.dot(xrand)-d.py*A_orig.dot(e.py*xrand)) <= TEST_EPS)



	PRINT("row {}-norms A:".format(PNORM))
	for i in xrange(m1):
		PRINT(i, ": ", np.linalg.norm(A_out.py[i,:],PNORM))
	PRINT("column {}-norms A:".format(PNORM))
	for j in xrange(n1):
		PRINT(j, ": ", np.linalg.norm(A_out.py[:,j],PNORM))


	PRINT("FAT MATRIX")


	if A_in is not None:
		B = Matrix(A_in.T)
		(m2,n2) = B.shape
	else:
		(m2,n2) = (30, 50)
		B = Matrix(np.random.rand(m2,n2))
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
		PRINT(i, ": ", np.linalg.norm(B.py[i,:],PNORM))
	PRINT("column {}-norms A:".format(PNORM))
	for j in xrange(n2):
		PRINT(j, ": ", np.linalg.norm(B.py[:,j],PNORM))

	PRINT("d: ", d.py)
	PRINT("e: ", e.py)
	equil_method(B,B_out,d,e)


	# equilibration should only change output matrix
	assert reduce(and_,map(lambda x: x==0, [dij for \
										di in B_orig-B.py for dij in di]))


	# verify B_out = DBE
	xrand = np.random.rand(n2)
	assert all((B_out.py.dot(xrand)-d.py*B_orig.dot(e.py*xrand)) <= TEST_EPS)



	PRINT("\AFTER")

	PRINT("row norms A:")
	for i in xrange(m2):
		PRINT(i, ": ", np.linalg.norm(B_out.py[i,:]))
	PRINT("column norms A:")
	for j in xrange(n2):
		PRINT(j, ": ", np.linalg.norm(B_out.py[:,j]))

	PRINT("d: ", d.py)
	PRINT("e: ", e.py)


	return True


def test_equil(*args,**kwargs):
	print "EQUILIBRATION TESTING\n\n\n\n\n"
	
	verbose = '--verbose' in args
	A = np.load(kwargs['file']) if 'file' in kwargs else None

	print "\n\nDENSE EQUIL"
	print "===========\n\n"
	assert dense_equil_test(dense_l2_equilibration,A_in=A,VERBOSE_TEST=verbose)
	print "...passed"

	print "\n\nSINKHORN KNOPP"
	print "==============\n\n"
	assert dense_equil_test(sinkhornknopp_equilibration,A_in=A,VERBOSE_TEST=verbose)
	print "...passed"
	return True

