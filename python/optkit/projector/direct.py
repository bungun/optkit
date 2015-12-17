from optkit.projector.base import Projector
from optkit.utils.pyutils import var_assert

class DirectProjectorFactory(object):
	def __init__(self, kernels, matrix_type):
		copy = kernels['copy']
		sync = kernels['sync']
		axpy = kernels['axpy']
		div = kernels['div']
		gemv = kernels['gemv']
		add_diag = kernels['add_diag']
		mean_diag = kernels['mean_diag']
		gramian = kernels['gramian']
		cholesky_factor = kernels['cholesky_factor']
		cholesky_solve = kernels['cholesky_solve']

		# forms chol(I+A'A) or chol(I+AA')
		def make_projector(A, normalize=True):
			L = gramian(A) # L = A'A or AA'
			sync(L)	

			normA = 1.
			mean_diagonal = mean_diag(L)
			if normalize: 
				div(mean_diagonal,L)
				div(mean_diagonal**0.5,A)
			normA = mean_diagonal**0.5

			add_diag(1,L)
			cholesky_factor(L)
			return L, normA

		class DirectProjector(Projector):
			def __init__(self, A, L=None, normalize=False):
				self.Matrix = matrix_type
				self.A=A
				if L is None:
					L, normA = make_projector(A, normalize=normalize)
				else:
					normA = 1.
				self.L = L
				self.normA = normA
				self.normalized = normalize
				
			def __call__(self, x, y, x_out, y_out):
				if self.A.skinny:
					# x = cholsolve(c + A^Td)
					# y = Ax
					copy(x,x_out)
					gemv('T', 1, self.A , y, 1, x_out)
					cholesky_solve(self.L, x_out)
					gemv('N', 1, self.A, x_out, 0, y_out)

				else:
					# y = d + cholsolve(Ac-d)
					# x = c - A^T(d-y)
					copy(y,y_out)
					gemv('N', 1, self.A, x, -1, y_out)
					cholesky_solve(self.L, y_out)
					gemv('T', -1, self.A, y_out, 0, x_out)
					axpy(1, y, y_out)
					axpy(1, x, x_out)

			def __str__(self):
				return str(self.__dict__)

			def isvalid(self):
				assert var_assert(self.A, self.L, self.Matrix)
				assert self.A.mindim == self.L.size1
				assert self.A.mindim == self.L.size2		
				assert isinstance(self.normA,float)
				assert isinstance(self.normalized,(bool,int))
				return True

		self.DirectProjector = DirectProjector

