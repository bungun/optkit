import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_void_p, byref, CFUNCTYPE
from optkit.libs import OperatorLibs
from optkit.tests.C.base import OptkitCTestCase

class OperatorLibsTestCase(OptkitCTestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = OperatorLibs()
		self.A_test = self.A_test_gen
		self.A_test_sparse = self.A_test_sparse_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])

	def tearDown(self):
		self.free_all_vars()

	def validate_operator(self, operator_, m, n, OPERATOR_KIND):
		o = operator_
		self.assertEqual(o.size1, m)
		self.assertEqual(o.size2, n)
		self.assertEqual(o.kind, OPERATOR_KIND)
		self.assertNotEqual(o.data, 0)
		self.assertNotEqual(o.apply, 0)
		self.assertNotEqual(o.adjoint, 0)
		self.assertNotEqual(o.fused_apply, 0)
		self.assertNotEqual(o.fused_adjoint, 0)
		self.assertNotEqual(o.free, 0)

	def exercise_operator(self, lib, operator_, A_py, TOL):
		o = operator_
		m, n = A_py.shape
		RTOL = TOL
		ATOLM = TOL * m**0.5
		ATOLN = TOL * n**0.5

		alpha = np.random.rand()
		beta = np.random.rand()

		# allocate vectors x, y
		x_ = np.zeros(n).astype(lib.pyfloat)
		x_ += self.x_test
		x_ptr = x_.ctypes.data_as(lib.ok_float_p)
		x = lib.vector(0, 0, None)
		self.assertCall( lib.vector_calloc(x, n) )
		self.register_var('x', x, lib.vector_free)

		self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

		y_ = np.zeros(m).astype(lib.pyfloat)
		y_ptr = y_.ctypes.data_as(lib.ok_float_p)
		y = lib.vector(0, 0, None)
		self.assertCall( lib.vector_calloc(y, m) )
		self.register_var('y', y, lib.vector_free)

		# test Ax
		Ax = A_py.dot(x_)
		self.assertCall( o.apply(o.data, x, y) )
		self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
		self.assertTrue( np.linalg.norm(y_ - Ax) <=
						 ATOLM + RTOL * np.linalg.norm(Ax) )

		# test A'y
		y_[:] = Ax[:] 	# (update for consistency)
		Aty = A_py.T.dot(y_)
		self.assertCall( o.adjoint(o.data, y, x) )
		self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
		self.assertTrue( np.linalg.norm(x_ - Aty) <=
						 ATOLN + RTOL * np.linalg.norm(Aty) )

		# test Axpy
		x_[:]  = Aty[:] # (update for consistency)
		Axpy = alpha * A_py.dot(x_) + beta * y_
		self.assertCall( o.fused_apply(o.data, alpha, x, beta, y) )
		self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
		self.assertTrue( np.linalg.norm(y_ - Axpy) <=
						 ATOLM + RTOL * np.linalg.norm(Axpy) )

		# test A'ypx
		y_[:] = Axpy[:] # (update for consistency)
		Atypx = alpha * A_py.T.dot(y_) + beta * x_
		self.assertCall( o.fused_adjoint(o.data, alpha, y, beta, x) )
		self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
		self.assertTrue( np.linalg.norm(x_ - Atypx) <=
						 ATOLN + RTOL * np.linalg.norm(Atypx) )

		self.free_var('x')
		self.free_var('y')

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self. libs.get(single_precision=single_precision,
									   gpu=gpu))
		self.assertTrue(any(libs))

	def test_dense_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				A = lib.matrix(0, 0, 0, None, order)
				self.assertCall( lib.matrix_calloc(A, m, n, order) )
				self.register_var('A', A, lib.matrix_free)

				o = lib.dense_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)
				self.validate_operator(o.contents, m, n, lib.enums.DENSE)

				self.free_var('o')
				self.free_var('A')

				self.assertCall( lib.ok_device_reset() )

	def test_dense_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_ = np.zeros(self.shape, order=pyorder).astype(lib.pyfloat)
				A_ += self.A_test
				A_ptr = A_.ctypes.data_as(lib.ok_float_p)
				A = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(A, m, n, order)
				self.register_var('A', A, lib.matrix_free)

				lib.matrix_memcpy_ma(A, A_ptr, order)

				o = lib.dense_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)

				self.exercise_operator(lib, o.contents, A_, TOL)

				self.free_var('o')
				self.free_var('A')

				self.assertCall( lib.ok_device_reset() )

	def test_sparse_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				spmat = csr_matrix if rowmajor else csc_matrix
				enum = lib.enums.SPARSE_CSR if rowmajor else \
					   lib.enums.SPARSE_CSC

				A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
				lib.sp_matrix_calloc(A, m, n, self.nnz, order)
				self.register_var('A', A, lib.sp_matrix_free)

				o = lib.sparse_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)
				self.validate_operator(o.contents, m, n, enum)

				self.free_var('o')
				self.free_var('A')

				self.assertCall( lib.ok_device_reset() )

	def test_sparse_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				spmat = csr_matrix if rowmajor else csc_matrix

				hdl = c_void_p()
				self.assertEqual(lib.sp_make_handle(byref(hdl)), 0)

				A_ = np.zeros(self.shape).astype(lib.pyfloat)
				A_ += self.A_test_sparse
				A_sp = spmat(A_)
				A_val = A_sp.data.ctypes.data_as(lib.ok_float_p)
				A_ind = A_sp.indices.ctypes.data_as(lib.ok_int_p)
				A_ptr = A_sp.indptr.ctypes.data_as(lib.ok_int_p)

				A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
				self.assertCall( lib.sp_matrix_calloc(A, m, n, A_sp.nnz,
													  order) )
				self.register_var('A', A, lib.sp_matrix_free)

				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val, A_ind,
														 A_ptr, order) )

				o = lib.sparse_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)

				self.exercise_operator(lib, o.contents, A_, TOL)

				self.free_var('o')
				self.free_var('A')

				self.assertEqual(lib.sp_destroy_handle(hdl), 0)
				self.assertCall( lib.ok_device_reset() )

	def test_diagonal_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for rowmajor in (True, False):
				d = lib.vector(0, 0, None)
				self.assertCall( lib.vector_calloc(d, n) )
				self.register_var('d', d, lib.vector_free)

				o = lib.diagonal_operator_alloc(d)
				self.register_var('o', o.contents.data, o.contents.free)

				self.validate_operator(o.contents, n, n, lib.enums.DIAGONAL)

				self.free_var('o')
				self.free_var('d')

				self.assertCall( lib.ok_device_reset() )

	def test_diagonal_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for rowmajor in (True, False):
				d_ = np.zeros(n).astype(lib.pyfloat)
				d_ += self.A_test[0, :]
				d_ptr = d_.ctypes.data_as(lib.ok_float_p)
				d = lib.vector(0, 0, None)
				self.assertCall( lib.vector_calloc(d, n) )
				self.register_var('d', d, lib.vector_free)

				self.assertCall( lib.vector_memcpy_va(d, d_ptr, 1) )

				o = lib.diagonal_operator_alloc(d)
				self.register_var('o', o.contents.data, o.contents.free)

				self.exercise_operator(lib, o.contents, np.diag(d_), TOL)

				self.free_var('o')
				self.free_var('d')

				self.assertCall( lib.ok_device_reset() )