import numpy as np
from optkit.types import Matrix, Vector
from optkit.equilibration import *

def dense_equil_test(equil_method):
	print "SKINNY MATRIX"

	(m1,n1) = (50, 30)
	print "m: {}\tn: {}".format(m1,n1)

	A = Matrix(np.random.rand(m1,n1))
	A_out = Matrix(np.zeros_like(A.py))

	d = Vector(m1)
	e = Vector(n1)

	print "\nBEFORE"

	print "row norms A:"
	for i in xrange(m1):
		print i, ": ", np.linalg.norm(A.py[i,:])
	print "column norms A:"
	for j in xrange(n1):
		print j, ": ", np.linalg.norm(A.py[:,j])

	print "d: ", d.py
	print "e: ", e.py
	equil_method(A,A_out,d,e)
	print "\AFTER"

	print "row norms A:"
	for i in xrange(m1):
		print i, ": ", np.linalg.norm(A_out.py[i,:])
	print "column norms A:"
	for j in xrange(n1):
		print j, ": ", np.linalg.norm(A_out.py[:,j])


	print "d: ", d.py
	print "e: ", e.py


	print "FAT MATRIX"

	(m2,n2) = (30, 50)
	print "m: {}\tn: {}".format(m2,n2)

	B = Matrix(np.random.rand(m2,n2))
	B_out = Matrix(np.zeros_like(B.py))

	d = Vector(m2)
	e = Vector(n2)

	print "\nBEFORE"

	print "row norms A:"
	for i in xrange(m2):
		print i, ": ", np.linalg.norm(B.py[i,:])
	print "column norms A:"
	for j in xrange(n2):
		print j, ": ", np.linalg.norm(B.py[:,j])

	print "d: ", d.py
	print "e: ", e.py
	equil_method(B,B_out,d,e)
	print "\AFTER"

	print "row norms A:"
	for i in xrange(m2):
		print i, ": ", np.linalg.norm(B_out.py[i,:])
	print "column norms A:"
	for j in xrange(n2):
		print j, ": ", np.linalg.norm(B_out.py[:,j])

	print "d: ", d.py
	print "e: ", e.py



def test_equil():
	print "\n\nDENSE EQUIL"
	print "===========\n\n"
	dense_equil_test(dense_l2_equilibration)

	print "\n\nSINKHORN KNOPP"
	print "==============\n\n"
	dense_equil_test(sinkhornknopp_equilibration)


