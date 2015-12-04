from optkit.projector.base import Projector
from optkit.types import Matrix, Vector
from optkit.kernels.linsys import *
from optkit.utils.pyutils import var_assert
from toolz import curry
import numpy as np


# forms chol(I+A'A) or chol(I+AA')
def make_projector(A, normalize=True):
	L = gramian(A) # L = A'A or AA'

	normA = 1.
	mean_diag = asum(diag(L))/A.mindim
	if normalize: 
		div(mean_diag,L)
		div(mean_diag**0.5,A)
	normA = mean_diag**0.5

	add_diag(1,L)
	cholesky_factor(L)
	return L, normA

@curry 
def direct_project(A, x, y, x_out, y_out):
	proj = DirectProjector(A)
	proj(x, y, x_out, y_out)
	

def factorize_and_project(A, x, y, x_out, y_out):
	proj = DirectProjector(A)
	proj(x, y, x_out, y_out)

class DirectProjector(Projector):
	def __init__(self, A, L=None, normalize=False):
		self.A=A
		if L is None: 
			L, normA = make_projector(A, normalize=normalize)
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
		assert var_assert(self.A,self.L,Matrix)
		assert self.A.mindim == self.L.size1
		assert self.A.mindim == self.L.size2		
		assert isinstance(self.normA,float)
		assert isinstance(self.normalized,(bool,int))
		return True




