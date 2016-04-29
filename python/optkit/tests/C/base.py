from numpy import zeros
from ctypes import c_void_p, c_size_t, byref
from scipy.sparse import csc_matrix, csr_matrix
from optkit.libs.error import optkit_print_error as PRINTERR
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

	def free_vars(self, *names):
		for name in names:
			self.free_var(name)

	def free_all_vars(self):
		for varname in self.managed_vars.keys():
			print 'releasing unfreed C variable {}'.format(varname)
			self.free_var(varname)

	def assertCall(self, call):
		self.assertEqual( PRINTERR(call), 0 )

	def register_vector(self, lib, size, name):
		if 'vector_calloc' in lib.__dict__:
			v = lib.vector(0, 0, None)
			self.assertCall( lib.vector_calloc(v, size) )
			self.register_var(name, v, lib.vector_free)
			v_py = zeros(size).astype(lib.pyfloat)
			v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
			return v, v_py, v_ptr
		else:
			raise ValueError('library {} cannot allocate a vector'.format(lib))

	def register_indvector(self, lib, size, name):
		if 'indvector_calloc' in lib.__dict__:
			v = lib.indvector(0, 0, None)
			self.assertCall( lib.indvector_calloc(v, size) )
			self.register_var(name, v, lib.indvector_free)
			v_py = zeros(size).astype(c_size_t)
			v_ptr = v_py.ctypes.data_as(lib.c_size_t_p)
			return v, v_py, v_ptr
		else:
			raise ValueError('library {} cannot allocate a vector'.format(lib))


	def register_matrix(self, lib, size1, size2, order, name):
		if 'matrix_calloc' in lib.__dict__:
			pyorder = 'C' if order == lib.enums.CblasRowMajor else 'F'

			A = lib.matrix(0, 0, 0, None, order)
			self.assertCall( lib.matrix_calloc(A, size1, size2, order) )
			self.register_var(name, A, lib.matrix_free)
			A_py = zeros((size1, size2), order=pyorder).astype(lib.pyfloat)
			A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
			return A, A_py, A_ptr
		else:
			raise ValueError('library {} cannot allocate a matrix'.format(lib))

	def register_sparsemat(self, lib, Adense, order, name):
		if 'sp_matrix_calloc' in lib.__dict__:

			A_py = zeros(Adense.shape).astype(lib.pyfloat)
			A_py += Adense
			A_sp = csr_matrix(A_py) if order == lib.enums.CblasRowMajor else \
				   csc_matrix(A_py)
			A_val = A_sp.data.ctypes.data_as(lib.ok_float_p)
			A_ind = A_sp.indices.ctypes.data_as(lib.ok_int_p)
			A_ptr = A_sp.indptr.ctypes.data_as(lib.ok_int_p)
			m, n = A_sp.shape
			nnz = A_sp.nnz

			A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
			self.assertCall( lib.sp_matrix_calloc(A, m, n, nnz, order) )
			self.register_var(name, A, lib.sp_matrix_free)
			return A, A_py, A_sp, A_val, A_ind, A_ptr

		else:
			raise ValueError(
					'library {} cannot allocate a sparse matrix'.format(lib))

	def register_blas_handle(self, lib, name):
		if 'blas_make_handle' in lib.__dict__:
			hdl = c_void_p()
			self.assertCall( lib.blas_make_handle(byref(hdl)) )
			self.register_var(name, hdl, lib.blas_destroy_handle)
			return hdl
		else:
			raise ValueError(
					'library {} cannot allocate a BLAS handle'.format(lib))

	def register_sparse_handle(self, lib, name):
		if 'sp_make_handle' in lib.__dict__:
			hdl = c_void_p()
			self.assertCall( lib.sp_make_handle(byref(hdl)) )
			self.register_var(name, hdl, lib.sp_destroy_handle)
			return hdl
		else:
			raise ValueError(
					'library {} cannot allocate a sparse handle'.format(lib))

	def register_fnvector(self, lib, size, name):
		if 'function_vector_calloc' in lib.__dict__:
			f_ = zeros(size, dtype=lib.function)
			f_ptr = f_.ctypes.data_as(lib.function_p)

			f = lib.function_vector(0, None)
			self.assertCall( lib.function_vector_calloc(f, size) )
			self.register_var(name, f, lib.function_vector_free)
			return f, f_, f_ptr

		else:
			raise ValueError(
					'library {} cannot allocate function vector'.format(lib))

class OptkitCOperatorTestCase(OptkitCTestCase):
	# A_test = None
	# A_test_sparse = None

	@property
	def op_keys(self):
		return ['dense', 'sparse']

	def register_operator(self, lib, opkey, rowmajor=True):
		if opkey == 'dense':
			return self.register_dense_operator(lib, self.A_test, rowmajor)
		elif opkey == 'sparse':
			return self.register_sparse_operator(lib, self.A_test_sparse, rowmajor)
		else:
			raise ValueError('invalid operator type')

	def register_dense_operator(self, lib, A_py, rowmajor=True):
		m, n = A_py.shape
		order = lib.enums.CblasRowMajor if rowmajor else \
				lib.enums.CblasColMajor
		A, A_, A_ptr = self.register_matrix(lib, m, n, order, 'A')
		A_ += A_py
		self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
		o = lib.dense_operator_alloc(A)
		self.register_var('o', o.contents.data, o.contents.free)
		return A_, A, o

	def register_sparse_operator(self, lib, A_py, rowmajor=True):
		m, n = A_py.shape
		order = lib.enums.CblasRowMajor if rowmajor else \
				lib.enums.CblasColMajor
		sparse_hdl = self.register_sparse_handle(lib, 'sp_hdl')
		A, A_, A_sp, A_val, A_ind, A_ptr = self.register_sparsemat(
			lib, A_py, order, 'A')
		lib.sp_matrix_memcpy_ma(sparse_hdl, A, A_val, A_ind, A_ptr)
		self.free_var('sp_hdl')
		o = lib.sparse_operator_alloc(A)
		self.register_var('o', o.contents.data, o.contents.free)
		return A_, A, o

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