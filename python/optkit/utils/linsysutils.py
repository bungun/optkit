from optkit.types import ok_enums as enums
from numpy import ndarray
from scipy.sparse import csr_matrix, csc_matrix

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


class UtilMakeCSparseMatrix(object):
	def __init__(self, sparse_handle, lowtypes, sparselib):
		self.sparse_handle = sparse_handle
		self.lowtypes = lowtypes
		self.sparselib = sparselib
		self.ndarray_pointer = lowtypes.ndarray_pointer

	def __call__(self, A=None):	

		if A is None:
			return self.lowtypes.sparse_matrix(0, 0, 0, 0, 
				None, None, None, enums.CblasRowMajor)
		elif isinstance(A, (csr_matrix, csc_matrix)):
			(m,n) = A.shape
			if self.lowtypes.order == 'col' and not isinstance(A, csc_matrix):
				A = csc_matrix(A)
			elif self.lowtypes.order == 'row' and not isinstance(A, csr_matrix):
				A = csr_matrix(A)

			order = enums.CblasRowMajor if isinstance(A, csr_matrix) else \
					enums.CblasColMajor

			A_ = self.lowtypes.sparse_matrix(0, 0, 0, 0, 
				None, None, None, order)
			
			self.sparselib.sp_matrix_calloc(A_, m, n, A.nnz, order)
			self.sparselib.sp_matrix_memcpy_ma(self.sparse_handle, A_, 
				self.ndarray_pointer(A.data),
				self.ndarray_pointer(A.indices), 
				self.ndarray_pointer(A.indptr), 
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

class UtilReleaseCSparseMatrix(object):
	def __init__(self, lowtypes, sparselib):
		self.lowtypes = lowtypes
		self.sparselib = sparselib
	def __call__(self, A):
		if isinstance(A, self.lowtypes.sparse_matrix):
			self.sparselib.sp_matrix_free(A)

# @curry
# def util release_cvector(lowtypes, denselib, x):
# 	if isinstance(x, self.lowtypes.vector):
# 		self.denselib.vector_free(x)

# @curry
# def util_release_cmatrix(lowtypes, denselib, A):
# 	if isinstance(A, self.lowtypes.matrix):
# 		self.denselib.matrix_free(A)

# @curry
# def util_release_csparsematrix(lowtypes, sparselib, A):
# 	if isinstance(A, self.lowtypes.sparse_matrix):
# 		self.sparselib.sp_matrix_free(A)
