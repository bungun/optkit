from optkit.kernels.linsys.core import *


def dense_equilibrate(A, d, e):
	m,n = A.shape
	set_all(1,e)
	for i in xrange(m):
		a = view(A,i,row=1)
		val = dot(a,a)
		if val == 0: raise ValueError("matrix A contains all-zero row")
		val **=-0.5
		d.py[i]=val
		mul(val,a)
	sync(d,python_to_C=True)









