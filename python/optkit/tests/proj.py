import numpy as np
from optkit.types import Matrix, Vector
from  optkit.projector import DirectProjector

def projector_test():

	print "\n\nDIRECT PROJECTOR"
	print "----------------\n\n"

	print "\nFAT MATRIX"
	print "----------\n"

	# fat matrix
	(m1,n1) = (1000, 3000)
	print "m: {}\tn: {}".format(m1,n1)

	A = Matrix(np.random.rand(m1,n1))
	ProjA = DirectProjector(A)

	x = Vector(np.random.rand(n1))
	y = Vector(np.random.rand(m1))
	x_out = Vector(n1)
	y_out = Vector(m1)

	print "RANDOM (x,y)"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x.py), np.linalg.norm(y.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y.py-A.py.dot(x.py))

	ProjA(x,y,x_out,y_out)

	print "PROJECT:"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x_out.py), np.linalg.norm(y_out.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y_out.py-A.py.dot(x_out.py))


	print "\nSKINNY MATRIX"
	print "-------------\n"

	# skinny matrix
	(m2,n2) = (3000, 1000)
	print "m: {}\tn: {}".format(m2,n2)

	B = Matrix(np.random.rand(m2,n2))
	ProjB = DirectProjector(B)


	x = Vector(np.random.rand(n2))
	y = Vector(np.random.rand(m2))
	x_out = Vector(n2)
	y_out = Vector(m2)

	print "RANDOM (x,y)"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x.py), np.linalg.norm(y.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y.py-B.py.dot(x.py))

	ProjB(x,y,x_out,y_out)


	print "PROJECT:"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x_out.py), np.linalg.norm(y_out.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y_out.py-B.py.dot(x_out.py))


	print "\n\nINDIRECT PROJECTOR"
	print "------------------\n\n"

	print "(NOT IMPLEMENTED)"


