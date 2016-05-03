import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_void_p, byref, CFUNCTYPE
from optkit.libs.operator import OperatorLibs
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
		self.exit_call()

	def validate_operator(self, operator_, m, n, OPERATOR_KIND):
		o = operator_
		self.assertEqual( o.size1, m )
		self.assertEqual( o.size2, n )
		self.assertEqual( o.kind, OPERATOR_KIND )
		self.assertNotEqual( o.data, 0 )
		self.assertNotEqual( o.apply, 0 )
		self.assertNotEqual( o.adjoint, 0 )
		self.assertNotEqual( o.fused_apply, 0 )
		self.assertNotEqual( o.fused_adjoint, 0 )
		self.assertNotEqual( o.free, 0 )

	def exercise_operator(self, lib, operator_, A_py, TOL):
		o = operator_
		m, n = A_py.shape
		RTOL = TOL
		ATOLM = TOL * m**0.5
		ATOLN = TOL * n**0.5

		alpha = np.random.rand()
		beta = np.random.rand()

		# allocate vectors x, y
		x, x_, x_ptr = self.register_vector(lib, n, 'x')
		y, y_, y_ptr = self.register_vector(lib, m, 'y')

		x_ += self.x_test
		self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

		# test Ax
		Ax = A_py.dot(x_)
		self.assertCall( o.apply(o.data, x, y) )
		self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
		self.assertVecEqual( y_, Ax, ATOLM, RTOL )

		# test A'y
		y_[:] = Ax[:] 	# (update for consistency)
		Aty = A_py.T.dot(y_)
		self.assertCall( o.adjoint(o.data, y, x) )
		self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
		self.assertVecEqual( x_, Aty, ATOLN, RTOL )

		# test Axpy
		x_[:]  = Aty[:] # (update for consistency)
		Axpy = alpha * A_py.dot(x_) + beta * y_
		self.assertCall( o.fused_apply(o.data, alpha, x, beta, y) )
		self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
		self.assertVecEqual( y_, Axpy, ATOLM, RTOL )

		# test A'ypx
		y_[:] = Axpy[:] # (update for consistency)
		Atypx = alpha * A_py.T.dot(y_) + beta * x_
		self.assertCall( o.fused_adjoint(o.data, alpha, y, beta, x) )
		self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
		self.assertVecEqual( x_, Atypx, ATOLN, RTOL )

		self.free_vars('x', 'y')
		self.assertCall( lib.ok_device_reset() )

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

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				A, _, _, = self.register_matrix(lib, m, n, order, 'A')

				o = lib.dense_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)
				self.validate_operator(o.contents, m, n, lib.enums.DENSE)

				self.free_vars('o', 'A')
				self.assertCall( lib.ok_device_reset() )

	def test_dense_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				A, A_, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				A_ += self.A_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				o = lib.dense_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)

				self.exercise_operator(lib, o.contents, A_, TOL)

				self.free_vars('o', 'A')
				self.assertCall( lib.ok_device_reset() )

	def test_sparse_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				enum = lib.enums.SPARSE_CSR if rowmajor else \
					   lib.enums.SPARSE_CSC

				A, _, _, _, _, _ = self.register_sparsemat(
						lib, self.A_test_sparse, order, 'A')

				o = lib.sparse_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)
				self.validate_operator(o.contents, m, n, enum)

				self.free_vars('o', 'A')
				self.assertCall( lib.ok_device_reset() )

	def test_sparse_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				A, A_, A_sp, A_val, A_ind, A_ptr = self.register_sparsemat(
						lib, self.A_test_sparse, order, 'A')

				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val, A_ind,
														 A_ptr, order) )

				o = lib.sparse_operator_alloc(A)
				self.register_var('o', o.contents.data, o.contents.free)

				self.exercise_operator(lib, o.contents, A_, TOL)

				self.free_vars('o', 'A', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_diagonal_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for rowmajor in (True, False):
				d = lib.vector(0, 0, None)
				self.assertCall( lib.vector_calloc(d, n) )
				self.register_var('d', d, lib.vector_free)

				o = lib.diagonal_operator_alloc(d)
				self.register_var('o', o.contents.data, o.contents.free)

				self.validate_operator(o.contents, n, n, lib.enums.DIAGONAL)

				self.free_vars('o', 'd')
				self.assertCall( lib.ok_device_reset() )

	def test_diagonal_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for rowmajor in (True, False):
				d, d_, d_ptr = self.register_vector(lib, n, 'd')
				d_ += self.A_test[0, :]
				self.assertCall( lib.vector_memcpy_va(d, d_ptr, 1) )

				o = lib.diagonal_operator_alloc(d)
				self.register_var('o', o.contents.data, o.contents.free)

				self.exercise_operator(lib, o.contents, np.diag(d_), TOL)

				self.free_vars('o', 'd')
				self.assertCall( lib.ok_device_reset() )