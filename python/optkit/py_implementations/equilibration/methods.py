from numpy import ndarray
import sys

class EquilibrationMethods(object):
	def __init__(self, kernels, vector_type, matrix_type):
		self.Vector = vector_type
		self.Matrix = matrix_type
		self.kernels = kernels


	def dense_l2(self, A, A_out, d, e):
		call = self.kernels
		try:
			assert isinstance(A, (ndarray, self.Matrix))
			assert isinstance(A_out, self.Matrix)
			assert isinstance(d, self.Vector)
			assert isinstance(e, self.Vector)
		except AssertionError:
			print sys.exc_info()[0]
			raise TypeError("Incorrect arguments to equilibration call",e)

		try:
			assert A.shape == A_out.shape
			assert (d.size,e.size) == A.shape
		except AssertionError:
			print sys.exc_info()[0]
			raise ValueError("Incompatible arguments to equilibration call",e)


		call['copy'](A,A_out)

		m,n = A_out.shape
		if not A_out.skinny:
			call['set_all'](1,e)
			for i in xrange(m):
				a = call['view'](A_out, i, row=1)
				val = call['dot'](a, a)
				if val == 0: raise ValueError("matrix A contains all-zero row")
				val **=-0.5
				d.py[i]=val
				call['mul'](val,a)
			call['sync'](d, python_to_C=True)
		else:
			call['set_all'](1,d)
			for j in xrange(n):
				a = call['view'](A_out, j, col=1)
				val = call['dot'](a, a)
				if val == 0: raise ValueError("matrix A contains all-zero column")
				val **=-0.5
				e.py[j]=val
				call['mul'](val,a)
			call['sync'](e, python_to_C=True)



	def sinkhornknopp(self, A, A_out, d, e):
		call = self.kernels
		try:
			assert isinstance(A, (ndarray, self.Matrix))
			assert isinstance(A_out, self.Matrix)
			assert isinstance(d, self.Vector)
			assert isinstance(e, self.Vector)
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

		call['copy'](A, A_out)
		call['set_all'](1, d)

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
			call['gemv']('T',1,A_out,d,0,e)
			call['sync'](e)

			e.py[:]**=-1
			call['sync'](e,python_to_C=True)

			call['gemv']('N', 1, A_out, e, 0, d)
			call['sync'](d)

			d.py[:]**=-1
			call['sync'](d, python_to_C=True)

			nrm_d = call['nrm2'](d) / sqrtm
			nrm_e = call['nrm2'](e) / sqrtn
			fac = (nrm_e / nrm_d)**0.5
			call['mul'](fac, d)
			call['div'](fac, e)

		call['sync'](d,e)
		# A_out = D*A*E
		call['copy'](A, A_out)
		for i in xrange(m):
			call['mul'](d.py[i], call['view'](A_out, i, row=1))
		for j in xrange(n):
			call['mul'](e.py[j], call['view'](A_out, j, col=1))