from collections import deque
from numpy import zeros, array, ndarray
from numpy.linalg import norm
from numpy.random import rand
from ctypes import c_void_p, c_size_t, byref
from scipy.sparse import csc_matrix, csr_matrix
from optkit.libs.error import optkit_print_error as PRINTERR
from optkit.tests.defs import OptkitTestCase

class OptkitCTestCase(OptkitTestCase):
	managed_vars = {}
	free_methods = {}
	var_stack = deque()
	libs = None

	@staticmethod
	def __exit_default():
		return None

	__exit_call = __exit_default

	def assertCall(self, call):
		self.assertEqual( PRINTERR(call), 0 )

	def assertVecEqual(self, first, second, atol, rtol):
		self.assertTrue( norm(first - second) <= atol + rtol * norm(second) )

	def assertVecNotEqual(self, first, second, atol, rtol):
		self.assertFalse( norm(first - second) <= atol + rtol * norm(second) )

	def assertScalarEqual(self, first, second, tol):
		self.assertTrue( abs(first - second) <= tol + tol * abs(second) )

	def register_var(self, name, var, free):
		self.managed_vars[name] = var
		self.free_methods[name] = free
		self.var_stack.append(name)	# preserve order of variable registration

	def unregister_var(self, name):
		self.managed_vars.pop(name, None)
		self.free_methods.pop(name, None)

	def free_var(self, name):
		var = self.managed_vars.pop(name, None)
		free_method = self.free_methods.pop(name, None)
		if free_method is not None and var is not None:
			PRINTERR( free_method(var) )

	def free_vars(self, *names):
		for name in names:
			self.free_var(name)

	def free_all_vars(self):
		for i in xrange(len(self.var_stack)):
			varname = self.var_stack.pop() # free in reverse of order added
			if varname in self.managed_vars:
				print 'releasing unfreed C variable {}'.format(varname)
			self.free_var(varname)

	# register device_reset() call so GPU is reset when tests error
	def register_exit(self, call):
		self.__exit_call = call

	def exit_call(self):
		PRINTERR( self.__exit_call() )
		self.__exit_call = self.__exit_default

	@staticmethod
	def gen_py_vector(lib, size, random=False):
		if 'ok_float_p' not in lib.__dict__:
			raise ValueError(
				'symbol "ok_float_p" undefined in library {}'.format(lib))

		v_py = zeros(size).astype(lib.pyfloat)
		v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
		if random:
			v_py += rand(size)
		return v_py, v_ptr

	@staticmethod
	def gen_py_matrix(lib, size1, size2, order, random=False):
		if 'ok_float_p' not in lib.__dict__:
			raise ValueError(
				'symbol "ok_float_p" undefined in library {}'.format(lib))

		pyorder = 'C' if order == lib.enums.CblasRowMajor else 'F'
		A_py = zeros((size1, size2), order=pyorder).astype(lib.pyfloat)
		A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
		if random:
			A_py += rand(size1, size2)
		return A_py, A_ptr

	def register_vector(self, lib, size, name, random=False):
		if not 'vector_calloc' in lib.__dict__:
			raise ValueError('library {} cannot allocate a vector'.format(lib))

		v = lib.vector(0, 0, None)
		self.assertCall( lib.vector_calloc(v, size) )
		self.register_var(name, v, lib.vector_free)
		v_py, v_ptr = self.gen_py_vector(lib, size, random)
		if random:
			self.assertCall( lib.vector_memcpy_va(v, v_ptr, 1) )
		return v, v_py, v_ptr

	def register_indvector(self, lib, size, name):
		if not 'indvector_calloc' in lib.__dict__:
			raise ValueError('library {} cannot allocate a vector'.format(lib))

		v = lib.indvector(0, 0, None)
		self.assertCall( lib.indvector_calloc(v, size) )
		self.register_var(name, v, lib.indvector_free)
		v_py = zeros(size).astype(c_size_t)
		v_ptr = v_py.ctypes.data_as(lib.c_size_t_p)
		return v, v_py, v_ptr

	def register_matrix(self, lib, size1, size2, order, name, random=False):
		if not 'matrix_calloc' in lib.__dict__:
			raise ValueError('library {} cannot allocate a matrix'.format(lib))

		A = lib.matrix(0, 0, 0, None, order)
		self.assertCall( lib.matrix_calloc(A, size1, size2, order) )
		self.register_var(name, A, lib.matrix_free)
		A_py, A_ptr = self.gen_py_matrix(lib, size1, size2, order, random)
		if random:
			self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
		return A, A_py, A_ptr

	def register_sparsemat(self, lib, Adense, order, name):
		if not 'sp_matrix_calloc' in lib.__dict__:
			raise ValueError('library {} cannot allocate a sparse '
							 'matrix'.format(lib))

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

	def register_blas_handle(self, lib, name):
		if not 'blas_make_handle' in lib.__dict__:
			raise ValueError('library {} cannot allocate a BLAS '
							 'handle'.format(lib))

		hdl = c_void_p()
		self.assertCall( lib.blas_make_handle(byref(hdl)) )
		self.register_var(name, hdl, lib.blas_destroy_handle)
		return hdl

	def register_sparse_handle(self, lib, name):
		if not 'sp_make_handle' in lib.__dict__:
			raise ValueError('library {} cannot allocate a sparse '
							 'handle'.format(lib))

		hdl = c_void_p()
		self.assertCall( lib.sp_make_handle(byref(hdl)) )
		self.register_var(name, hdl, lib.sp_destroy_handle)
		return hdl

	def register_fnvector(self, lib, size, name):
		if not 'function_vector_calloc' in lib.__dict__:
			raise ValueError('library {} cannot allocate function '
							 'vector'.format(lib))

		f_ = zeros(size, dtype=lib.function)
		f_ptr = f_.ctypes.data_as(lib.function_p)

		f = lib.function_vector(0, None)
		self.assertCall( lib.function_vector_calloc(f, size) )
		self.register_var(name, f, lib.function_vector_free)
		return f, f_, f_ptr

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
