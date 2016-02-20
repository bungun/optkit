from toolz import curry

class LinsysExtensionKernels(object):
	def __init__(self, linsys_kernels, matrix_type):
		self.copy = linsys_kernels.copy
		self.view = linsys_kernels.view
		self.sync = linsys_kernels.sync
		self.add = linsys_kernels.add
		self.mul = linsys_kernels.mul
		self.dot = linsys_kernels.dot
		self.asum = linsys_kernels.asum
		self.axpy = linsys_kernels.axpy
		self.gemv = linsys_kernels.gemv
		self.gemm = linsys_kernels.gemm
		self.Matrix = matrix_type

	def splitview(self, z,dim_y,y_before_x=True):
		N = z.size
		if not y_before_x:
			x = self.view(z, (0, N-dim_y))
			y = self.view(z, (dim_y, N))
		else:
			y = self.view(z, (0, dim_y))
			x = self.view(z, (dim_y, N))
		return x,y

	def axpby(self, a,x,b,y):
		if b != 1: self.mul(b,y)
		self.axpy(a,x,y)

	def axpby_inplace(self, a, x, b, y, z):
		self.copy(y, z)
		self.axpby(a, x, b, z)

	def diag(self, A):
		return self.view(A, diag=1)

	def aidpm(self, alpha, beta, A):
		if beta != 1: self.mul(beta, A)
		if alpha != 0: self.add(alpha, self.diag(A))

	def add_diag(self, alpha, A):
		self.aidpm(alpha, 1, A)

	def sum_diag(self, alpha, A):
		return sum(self.diag(A).py)

	def mean_diag(self, A):
		return self.asum(self.diag(A))/A.mindim

	def norm_diag(self, A, norm=2):
		if norm==1:
			nd = self.asum(self.diag(A))
		else:
			dA = self.diag(A)
			nd = self.dot(dA,dA)
		return nd

	def gramian(self, A):
		# returns A'A or AA'
		(t1,t2) = ('T','N') if A.skinny else ('N','T')
		AA = self.Matrix(A.mindim, A.mindim)
		self.gemm(t1, t2, 1, A, A, 0, AA)
		self.sync(AA)
		return AA

	def get_curryable_gemv(self):
		@curry
		def gemv_curried(A, tA, alpha, x, beta, y):
			self.gemv(tA, alpha, A, x, beta, y)
		return gemv_curried