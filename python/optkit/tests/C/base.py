from numpy import zeros
from ctypes import c_void_p, byref
from scipy.sparse import csc_matrix, csr_matrix
from optkit.tests.defs import OptkitTestCase

class OptkitCTestCase(OptkitTestCase):
	managed_vars = {}
	free_methods = {}
	libs = None

	def register_var(self, name, var, free):
		self.managed_vars[name] = var;
		self.free_methods[name] = free;

	def unregister_var(self, name):
		self.managed_vars.pop(name, None)
		self.free_methods.pop(name, None)

	def free_var(self, name):
		var = self.managed_vars.pop(name, None)
		free_method = self.free_methods.pop(name, None)
		if free_method is not None and var is not None:
			free_method(var)

	def free_all_vars(self):
		for varname in self.managed_vars.keys():
			print 'releasing unfreed C variable {}'.format(varname)
			self.free_var(varname)

class OptkitCOperatorTestCase(OptkitCTestCase):
	A_test = None
	A_test_sparse = None

	@property
	def op_keys(self):
		return ['dense', 'sparse']

	def gen_operator(self, opkey, lib, rowmajor=True):
		if opkey == 'dense':
			return self.gen_dense_operator(lib, self.A_test, rowmajor)
		elif opkey == 'sparse':
			return self.gen_sparse_operator(lib, self.A_test_sparse, rowmajor)
		else:
			raise ValueError('invalid operator type')

	@staticmethod
	def gen_dense_operator(lib, A_py, rowmajor=True):
		m, n = A_py.shape
		order = lib.enums.CblasRowMajor if rowmajor else \
				lib.enums.CblasColMajor
		pyorder = 'C' if rowmajor else 'F'
		A_ = zeros(A_py.shape, order=pyorder).astype(lib.pyfloat)
		A_ += A_py
		A_ptr = A_.ctypes.data_as(lib.ok_float_p)
		A = lib.matrix(0, 0, 0, None, order)
		lib.matrix_calloc(A, m, n, order)
		lib.matrix_memcpy_ma(A, A_ptr, order)
		o = lib.dense_operator_alloc(A)
		return A_, A, o, lib.matrix_free

	@staticmethod
	def gen_sparse_operator(lib, A_py, rowmajor=True):
		m, n = A_py.shape
		order = lib.enums.CblasRowMajor if rowmajor else \
				lib.enums.CblasColMajor
		sparsemat = csr_matrix if rowmajor else csc_matrix
		sparse_hdl = c_void_p()
		lib.sp_make_handle(byref(sparse_hdl))
		A_ = A_py.astype(lib.pyfloat)
		A_sp = csr_matrix(A_)
		A_ptr = A_sp.indptr.ctypes.data_as(lib.ok_int_p)
		A_ind = A_sp.indices.ctypes.data_as(lib.ok_int_p)
		A_val = A_sp.data.ctypes.data_as(lib.ok_float_p)
		A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
		lib.sp_matrix_calloc(A, m, n, A_sp.nnz, order)
		lib.sp_matrix_memcpy_ma(sparse_hdl, A, A_val, A_ind, A_ptr)
		lib.sp_destroy_handle(sparse_hdl)
		o = lib.sparse_operator_alloc(A)
		return A_, A, o, lib.sp_matrix_free

class OptkitCPogsTestCase(OptkitCTestCase):
	class PogsVariablesLocal():
		def __init__(self, m, n, pytype):
			self.m = m
			self.n = n
			self.z = zeros(m + n).astype(pytype)
			self.z12 = zeros(m + n).astype(pytype)
			self.zt = zeros(m + n).astype(pytype)
			self.zt12 = zeros(m + n).astype(pytype)
			self.prev = zeros(m + n).astype(pytype)
			self.d = zeros(m).astype(pytype)
			self.e = zeros(n).astype(pytype)

		@property
		def x(self):
			return self.z[self.m:]

		@property
		def y(self):
			return self.z[:self.m]

		@property
		def x12(self):
			return self.z12[self.m:]

		@property
		def y12(self):
			return self.z12[:self.m]

		@property
		def xt(self):
			return self.zt[self.m:]

		@property
		def yt(self):
			return self.zt[:self.m]

		@property
		def xt12(self):
			return self.zt12[self.m:]

		@property
		def yt12(self):
			return self.zt12[:self.m]

	class PogsOutputLocal():
		def __init__(self, lib, m, n):
			self.x = zeros(n).astype(lib.pyfloat)
			self.y = zeros(m).astype(lib.pyfloat)
			self.mu = zeros(n).astype(lib.pyfloat)
			self.nu = zeros(m).astype(lib.pyfloat)
			self.ptr = lib.pogs_output(self.x.ctypes.data_as(lib.ok_float_p),
									   self.y.ctypes.data_as(lib.ok_float_p),
									   self.mu.ctypes.data_as(lib.ok_float_p),
									   self.nu.ctypes.data_as(lib.ok_float_p))