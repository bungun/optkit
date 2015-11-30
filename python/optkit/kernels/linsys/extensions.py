from optkit.types import Vector, Matrix
from optkit.kernels.linsys.core import *

def splitview(z,dim_y,y_before_x=True):
	N = z.size
	if not y_before_x:
		x = view(z, (0,N-dim_y))
		y = view(z, (dim_y,N))
	else:
		y = view(z, (0,dim_y))
		x = view(z, (dim_y,N))
	return x,y

def axpby(a,x,b,y):
	# if isinstance(x,Vector) and (y,Vector):
	# elif isinstance(x,BlockVector) and (y,BlockVector):
	if b != 1: mul(b,y)
	axpy(a,x,y)


def axpby_inplace(a,x,b,y,z):
	copy(y,z)
	axpby(a,x,b,z)

def diag(A):
	return view(A,diag=1)

def aidpm(alpha, beta, A):
	if beta != 1: scale(beta, A)
	if alpha != 0: add(alpha, diag(A))

def add_diag(alpha, A):
	aidpm(alpha, 1, A)

def sum_diag(alpha, A):
	return sum(diag(A).py)

def norm_diag(A, norm=2):
	if norm==1:
		nd = asum(diag(A))
	else:
		dA = diag(A)
		nd = dot(dA,dA)

# returns A'A or AA'
def gramian(A):
	(t1,t2) = ('T','N') if A.skinny else ('N','T')
	AA = Matrix(A.mindim,A.mindim)
	gemm(t1, t2, 1, A, A, 0, AA) 	
	return AA
	
def blockcopy(z_orig, z_dest):
	copy(z_orig.vec, z_dest.vec)

def blockdot(z1, z2):
	return dot(z1.vec, z2.vec)