import numpy as np
from optkit.types import Matrix, Vector
from optkit.kernels import copy
from optkit.projector import DirectProjector
from optkit.equilibration import *
from optkit.tests.proj import direct_proj_test



def test_normalizedprojector():
	print "\n============="
	print "BEGIN TESTING"
	print "============="

	print "\n\nDIRECT PROJECTOR"
	print "----------------\n\n"

	# fat matrix
	print "\nFAT MATRIX"
	print "----------\n"
	(m,n) = (1000, 3000)
	A = Matrix(np.random.rand(m,n))
	A_equil = Matrix(np.zeros_like(A.py))
	d = Vector(m)
	e = Vector(n)

	print "m: {}\tn: {}".format(m,n)
	print "\nORIGINAL MATRIX"
	direct_proj_test(m,n,A)

	copy(A,A_equil)
	print "\nORIGINAL MATRIX, normalized"
	direct_proj_test(m,n,A_equil,normalize=True)


	print "\nL2-EQUILIBRATED MATRIX"
	dense_l2_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil)

	print "\nL2-EQUILIBRATED MATRIX, normalized"
	dense_l2_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil,normalize=True)


	print "\nSINKHORN-EQUILIBRATED MATRIX"
	sinkhornknopp_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil)	

	print "\nSINKHORN-EQUILIBRATED MATRIX, normalized"
	sinkhornknopp_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil,normalize=True)	


	# skinny matrix
	print "\nSKINNY MATRIX"
	print "-------------\n"

	(m,n) = (n,m)
	A = Matrix(np.random.rand(m,n))
	A_equil = Matrix(np.zeros_like(A.py))
	d = Vector(m)
	e = Vector(n)

	print "m: {}\tn: {}".format(m,n)
	print "\nORIGINAL MATRIX"
	direct_proj_test(m,n,A)

	copy(A,A_equil)
	print "\nORIGINAL MATRIX, normalized"
	direct_proj_test(m,n,A_equil,normalize=True)


	print "\nL2-EQUILIBRATED MATRIX"
	dense_l2_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil)

	print "\nL2-EQUILIBRATED MATRIX, normalized"
	dense_l2_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil,normalize=True)


	print "\nSINKHORN-EQUILIBRATED MATRIX"
	sinkhornknopp_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil)	

	print "\nSINKHORN-EQUILIBRATED MATRIX, normalized"
	sinkhornknopp_equilibration(A,A_equil,d,e)
	direct_proj_test(m,n,A_equil,normalize=True)	


	print "\n\nINDIRECT PROJECTOR"
	print "------------------\n\n"

	print "(NOT IMPLEMENTED)"

	print "\n==========="
	print "END TESTING"
	print "==========="
