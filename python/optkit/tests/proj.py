import numpy as np
from optkit.types import Matrix, Vector
from  optkit.projector import DirectProjector

def direct_proj_test(m,n,A=None,normalize=False):
	A = Matrix(np.random.rand(m,n)) if A is None else A
	ProjA = DirectProjector(A,normalize=normalize)

	x = Vector(np.random.rand(n))
	y = Vector(np.random.rand(m))
	x_out = Vector(n)
	y_out = Vector(m)

	print "RANDOM (x,y)"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x.py), np.linalg.norm(y.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y.py-A.py.dot(x.py))

	print "NORM A (from projector): ", ProjA.normA
	ProjA(x,y,x_out,y_out)

	print "PROJECT:"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x_out.py), np.linalg.norm(y_out.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y_out.py-A.py.dot(x_out.py))



def test_projector():

	print "\n============="
	print "BEGIN TESTING"
	print "============="

	print "\n\nDIRECT PROJECTOR"
	print "----------------\n\n"

	# fat matrix
	print "\nFAT MATRIX"
	print "----------\n"
	(m,n) = (1000, 3000)
	print "m: {}\tn: {}".format(m,n)
	direct_proj_test(m,n)

	# skinny matrix
	print "\nSKINNY MATRIX"
	print "-------------\n"
	print "m: {}\tn: {}".format(n,m)
	direct_proj_test(n,m)


	print "\n\nINDIRECT PROJECTOR"
	print "------------------\n\n"

	print "(NOT IMPLEMENTED)"


	print "\n==========="
	print "END TESTING"
	print "==========="