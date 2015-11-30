from optkit import Matrix, FunctionVector, pogs
import numpy as np
from operator import add

def pogs_test(*size, **kwargs):
	HLINE  = reduce(add, ['-' for i in xrange(100)]) + "\n"

	if len(size)==2:
		(m,n)=int(size[0]),int(size[1])
	else:
		(m,n)=(300,200)

	A = Matrix(np.random.rand(m,n))
	f = FunctionVector(m, h='Abs', b=1)
	g = FunctionVector(n, h='IndGe0')
	info, output, solver_state = pogs(A,f,g)

	print HLINE, HLINE
	print "INFO:"
	print info
	
	print HLINE, HLINE
	print "OUTPUT:"
	print output

	print HLINE, HLINE
	print "SOLVER STATE:"
	print solver_state
