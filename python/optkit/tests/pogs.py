from optkit import Matrix, FunctionVector, pogs
import numpy as np
from operator import add
from numpy.linalg import norm

def test_pogs(*size, **kwargs):
	HLINE  = reduce(add, ['-' for i in xrange(100)]) + "\n"

	print HLINE, HLINE, HLINE
	print "SKINNY MATRIX"

	if len(size)==2:
		(m,n)=int(size[0]),int(size[1])
		(m,n)=(max(m,n),min(m,n))
	else:
		(m,n)=(300,200)

	print "m: {}, n: {}".format(m,n)


	A_ = np.random.rand(m,n)
	A = Matrix(np.copy(A_))

	f = FunctionVector(m, h='Abs', b=1)
	g = FunctionVector(n, h='IndGe0')
	info, output, solver_state = pogs(A,f,g,verbose=1)

	print HLINE
	print "PRIMAL & DUAL FEASIBILITY"
	print "||Ax-y||: ", np.linalg.norm(A_.dot(output.x)-output.y)
	print "||A'nu+mu||: ", np.linalg.norm(A_.T.dot(output.nu)+output.mu)

	print HLINE, HLINE
	print "INFO:"
	print info
	
	print HLINE, HLINE
	print "OUTPUT:"
	print output

	print HLINE, HLINE
	print "SOLVER STATE:"
	print solver_state

	print HLINE, HLINE, HLINE
	print "FAT MATRIX"

	(m,n) = (n,m)
	print "m: {}, n: {}".format(m,n)

	A_ = np.random.rand(m,n)
	A = Matrix(np.copy(A_))

	f = FunctionVector(m, h='Abs', b=1)
	g = FunctionVector(n, h='IndGe0')
	info, output, solver_state = pogs(A,f,g,verbose=1)

	print HLINE
	print "PRIMAL & DUAL FEASIBILITY"
	print "||Ax-y||: ", np.linalg.norm(A_.dot(output.x)-output.y)
	print "||A'nu+mu||: ", np.linalg.norm(A_.T.dot(output.nu)+output.mu)

	print HLINE, HLINE
	print "INFO:"
	print info
	
	print HLINE, HLINE
	print "OUTPUT:"
	print output

	print HLINE, HLINE
	print "SOLVER STATE:"
	print solver_state
