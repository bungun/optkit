from optkit.kernels.linsys.core import *
from optkit.types import Vector,Matrix
import numpy as np
import sys 

def dense_l2_equilibration(A, A_out, d, e):
	try:
		assert isinstance(A, (np.ndarray,Matrix))
		assert isinstance(A_out,Matrix)
		assert isinstance(d,Vector)
		assert isinstance(e,Vector)
	except AssertionError:
		print sys.exc_info()[0]
		raise TypeError("Incorrect arguments to equilibration call",e)

	try:
		assert A.shape == A_out.shape
		assert (d.size,e.size) == A.shape
	except AssertionError:
		print sys.exc_info()[0]
		raise ValueError("Incompatible arguments to equilibration call",e)


	copy(A,A_out)

	m,n = A_out.shape
	if not A_out.skinny:
		set_all(1,e)
		for i in xrange(m):
			a = view(A_out,i,row=1)
			val = dot(a,a)
			if val == 0: raise ValueError("matrix A contains all-zero row")
			val **=-0.5
			d.py[i]=val
			mul(val,a)
		sync(d,python_to_C=True)
	else:
		set_all(1,d)
		for j in xrange(n):
			a = view(A_out,j,col=1)
			val=dot(a,a)
			if val == 0: raise ValueError("matrix A contains all-zero column")
			val **=-0.5
			e.py[j]=val
			mul(val,a)



def sinkhornknopp_equilibration(A, A_out, d, e):
	try:
		assert isinstance(A, (np.ndarray,Matrix))
		assert isinstance(A_out,Matrix)
		assert isinstance(d,Vector)
		assert isinstance(e,Vector)
	except AssertionError:
		print sys.exc_info()[0]
		raise TypeError("Incorrect arguments to equilibration call",e)

	try:
		assert A.shape == A_out.shape
		assert (d.size,e.size) == A.shape
	except AssertionError:
		print sys.exc_info()[0]
		raise ValueError("Incompatible arguments to equilibration call",e)

	NUMITER=10
	(m,n) = A_out.shape

	copy(A, A_out)
	set_all(1,d)



	sqrtm = m**0.5
	sqrtn = n**0.5


	# A_out = |A|
	for i in xrange(A_out.size1):
		for j in xrange(A_out.size2):
			A_out.py[i,j]=abs(A_out.py[i,j])

	# repeat NUMITER times
	# e = 1./ A_out^T d;
	# d = 1./ A_out e
	for k in xrange(NUMITER):
		gemv('T',1,A_out,d,0,e)

		e.py[:]**=-1
		sync(e,python_to_C=True)

		gemv('N',1,A_out,e,0,d)

		d.py[:]**=-1
		sync(d,python_to_C=True)

		nrm_d = nrm2(d) / sqrtm
		nrm_e = nrm2(e) / sqrtn
		fac = (nrm_e/nrm_d)**0.5
		mul(fac,d)
		div(fac,e)

	# A_out = D*A*E
	copy(A, A_out)
	for i in xrange(m):
		mul(d.py[i],view(A_out,i,row=1))
	for j in xrange(n):
		mul(e.py[j],view(A_out,j,col=1))



