# from optkit.compat import *

# import numpy as np
# import scipy.sparse as sp
# import ctypes as ct

# class OperatorTypes(object):
# 	def __init__(self, backend):
# 		pogslib = backend.pogs
# 		denselib = backend.dense
# 		sparselib = backend.sparse
# 		operatorlib = backend.operator

# 		accepted_operators = [str(np.ndarray), str(np.csr_matrix), str(np.csc_matrix),
# 							  str(np.coo_matrix)]

# 		def AbstractLinearOperator(object):
# 			def __init__(self, py_operator):
# 				self.py = py_operator
# 				self.__c_data = None
# 				self.c_ptr = None
# 				self.__free_data = lambda op : None
# 				self.shape = None

# 				input_is_sparse = isinstance(py_operator, (np.csr_matrix,
# 											 np.csc_matrix, np.coo_matrix))

# 				if isinstance(py_operator, np.ndarray):
# 					if len(py_operator.shape) != 2:
# 						raise ValueError('argument "py_operator" must be a 2-D'
# 										 '{} when provided as a {}'.format(
# 										 np.ndarray, np.ndarray))

# 					m, n = self.shape = self.py.shape
# 					order = denselib.enums.CblasRowMajor if \
# 							self.py.flags.c_contiguous else \
# 							denselib.enums.CblasColMajor
# 					input_ = self.py.astype(denselib.pyfloat)
# 					input_ptr = input_.ctypes.data_as(denselib.ok_float_p)

# 					self.__c_data = denselib.matrix(0, 0, 0, None, 0)
# 					denselib.matrix_calloc(self.__c_data, m, n, order)
# 					backend.increment_cobject_count()

# 					denselib.matrix_memcpy_ma(self.__c_data, input_ptr)

# 					self.c_ptr = operatorlib.dense_operator_alloc(
# 							self.__c_data)
# 					backend.increment_cobject_count()

# 					self.__free_data = denselib.matrix_free

# 				elif input_is_sparse:
# 					if isinstance(py_operator, np.coo_matrix):
# 						print "sparse matrix: converting COO input to CSR"
# 						self.py = np.csr_matrix(self.py)

# 					m, n = self.shape = self.py.shape

# 					hdl = ct.c_void_p()
# 					denselib.blas_make_handle(ct.byref(hdl))
# 					self.__c_data = sparselib.sparse_matrix(
# 							0, 0, 0, 0, None, None, None, 0)
# 					A_ptr_p = self.py.indptr.ctypes.data_as(denselib.ok_int_p)
# 					A_ind_p = self.py.indices.ctypes.data_as(denselib.ok_int_p)
# 					A_val_p = self.py.data.ctypes.data_as(denselib.ok_float_p)
# 					sparselib.sp_matrix_calloc(
# 							self.__c_data, m, n, self.py.nnz, order)
# 					backend.increment_cobject_count()

# 					sparselib.sp_matrix_memcpy_ma(
# 							hdl, self.__c_data, A_val_p, A_ind_p, A_ptr_p)
# 					denselib.blas_destroy_handle(hdl)

# 					self.c_ptr = operatorlib.sparse_operator_alloc(
# 							self.__c_data)
# 					backend.increment_cobject_count()

# 					self.__free_data = sparselib.sp_matrix_free

# 				else
# 					raise TypeError('argument "py_operator" must be one of '
# 									'{}'.format(accepted_operators))

# 			def __del__(self):
# 				if isinstance(self.c_ptr, operatorlib.operator_p):
# 					self.c_ptr.contents.free(self.c_ptr.contents.data)
# 					backend.decrement_cobject_count()

# 				if self.__c_data is not None:
# 					self.__free_data(self.__c_data)
# 					backend.decrement_cobject_count()

# 			self.AbstractLinearOperator = AbstractLinearOperator