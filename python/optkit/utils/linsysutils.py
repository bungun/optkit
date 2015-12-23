from optkit.types import ok_enums as enums
from numpy import ndarray

# low-level utilities
class UtilMakeCVector(object):
	def __init__(self, lowtypes, denselib):
		self.lowtypes = lowtypes
		self.denselib = denselib
		self.ndarray_pointer = lowtypes.ndarray_pointer
	def __call__(self, x=None, copy_data=True):
		if x is None:
			return self.lowtypes.vector(0,0,None)
		elif isinstance(x, ndarray) and len(x.shape)==1:
			x_ = self.lowtypes.vector(0,0,None)
			if not copy_data:
				self.denselib.vector_view_array(x_, self.ndarray_pointer(x), x.size)
			else:
				self.denselib.vector_calloc(x_, x.size)
				self.denselib.vector_memcpy_va(x_, self.ndarray_pointer(x), 1)	 
			return x_
		else:
			return None
			# TODO: error message (type, dims)

class UtilMakeCMatrix(object):
	def __init__(self, lowtypes, denselib):
		self.lowtypes = lowtypes
		self.denselib = denselib
		self.ndarray_pointer = lowtypes.ndarray_pointer
	def __call__(self, A=None, copy_data=True):	

		if A is None:
			return self.lowtypes.matrix(0,0,0,None,enums.CblasRowMajor)
		elif isinstance(A, ndarray) and len(A.shape)==2:
			(m,n) = A.shape
			if self.lowtypes.order == 'col' and not A.flags.f_contiguous:
				A_ = ndarray(shape=(m,n), order='F')
				A_[:] = A[:]
				A = A_
			elif self.lowtypes.order == 'row' and not A.flags.c_contiguous:
				A = ndarray(shape=(m,n), order='C')
				A_[:] = A[:]
				A = A_

			order = enums.CblasRowMajor if A.flags.c_contiguous else \
					enums.CblasColMajor

			A_ = self.lowtypes.matrix(0,0,0,None,order)
			if not copy_data:
				self.denselib.matrix_view_array(A_, self.ndarray_pointer(A), 
					m, n, order)
			else:
				self.denselib.matrix_calloc(A_, m, n, order)
				self.denselib.matrix_memcpy_ma(A_, self.ndarray_pointer(A), 
					order)
			return A_
		else:
			return None
			# TODO: error message (type, dims)

class UtilReleaseCVector(object):
	def __init__(self, lowtypes, denselib):
		self.lowtypes = lowtypes
		self.denselib = denselib
	def __call__(self, x):
		if isinstance(x, self.lowtypes.vector):
			self.denselib.vector_free(x)

class UtilReleaseCMatrix(object):
	def __init__(self, lowtypes, denselib):
		self.lowtypes = lowtypes
		self.denselib = denselib
	def __call__(self, A):
		if isinstance(A, self.lowtypes.matrix):
			self.denselib.matrix_free(A)