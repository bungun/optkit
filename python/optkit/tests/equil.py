import numpy as np
from optkit.types import Matrix, Vector
from optkit.utils.pyutils import println,printvoid, var_assert
from optkit.equilibration import *

def dense_equil_test(equil_method,VERBOSE_TEST=True):
	PRINT = println if VERBOSE_TEST else printvoid

	PRINT("SKINNY MATRIX")

	(m1,n1) = (50, 30)
	PRINT("m: {}\tn: {}".format(m1,n1))

	A = Matrix(np.random.rand(m1,n1))
	A_out = Matrix(np.zeros_like(A.py))

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

	PRINT("row norms A:")
	for i in xrange(m1):
		PRINT(i, ": ", np.linalg.norm(A_out.py[i,:]))
	PRINT("column norms A:")
	for j in xrange(n1):
		PRINT(j, ": ", np.linalg.norm(A_out.py[:,j]))


	PRINT("d: ", d.py)
	PRINT("e: ", e.py)

	# allow 30% variation in row and column norms
	row_norms=[np.linalg.norm(A_out.py[i,:]) for i in xrange(A_out.shape[0])]
	col_norms=[np.linalg.norm(A_out.py[:,j]) for j in xrange(A_out.shape[1])]
	assert np.min(row_norms)/np.max(row_norms) >= 0.5
	assert np.min(col_norms)/np.max(col_norms) >= 0.5


	PRINT("FAT MATRIX")

	(m2,n2) = (30, 50)
	PRINT("m: {}\tn: {}".format(m2,n2))

	B = Matrix(np.random.rand(m2,n2))
	B_out = Matrix(np.zeros_like(B.py))

	d = Vector(m2)
	e = Vector(n2)

	assert var_assert(B,B_out,type=Matrix)
	assert var_assert(d,e,type=Vector)


	PRINT("\nBEFORE")

	PRINT("row norms A:")
	for i in xrange(m2):
		PRINT(i, ": ", np.linalg.norm(B.py[i,:]))
	PRINT("column norms A:")
	for j in xrange(n2):
		PRINT(j, ": ", np.linalg.norm(B.py[:,j]))

	PRINT("d: ", d.py)
	PRINT("e: ", e.py)
	equil_method(B,B_out,d,e)
	PRINT("\AFTER")

	PRINT("row norms A:")
	for i in xrange(m2):
		PRINT(i, ": ", np.linalg.norm(B_out.py[i,:]))
	PRINT("column norms A:")
	for j in xrange(n2):
		PRINT(j, ": ", np.linalg.norm(B_out.py[:,j]))

	PRINT("d: ", d.py)
	PRINT("e: ", e.py)

	# allow 30% variation in row and column norms
	row_norms=[np.linalg.norm(B_out.py[i,:]) for i in xrange(B_out.shape[0])]
	col_norms=[np.linalg.norm(B_out.py[:,j]) for j in xrange(B_out.shape[1])]
	assert np.min(row_norms)/np.max(row_norms) >= 0.5
	assert np.min(col_norms)/np.max(col_norms) >= 0.5
	return True



def test_equil(*args):
	print "EQUILIBRATION TESTING\n\n\n\n\n"
	
	verbose = '--verbose' in args
	print "\n\nDENSE EQUIL"
	print "===========\n\n"
	assert dense_equil_test(dense_l2_equilibration,VERBOSE_TEST=verbose)
	print "...passed"

	print "\n\nSINKHORN KNOPP"
	print "==============\n\n"
	assert dense_equil_test(sinkhornknopp_equilibration,VERBOSE_TEST=verbose)
	print "...passed"
	return True

