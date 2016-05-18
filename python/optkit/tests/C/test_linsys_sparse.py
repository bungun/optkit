import os
import numpy as np
import scipy.sparse as sp
from ctypes import c_int, c_uint, Structure, byref, c_void_p
from optkit.libs.linsys import SparseLinsysLibs
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.base import OptkitCTestCase

class SparseLibsTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = SparseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue( any(libs) )

	def test_lib_types(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertTrue('ok_int_p' in dir(lib))
			self.assertTrue('ok_float_p' in dir(lib))
			self.assertTrue('sparse_matrix' in dir(lib))
			self.assertTrue('sparse_matrix_p' in dir(lib))

	def test_sparse_handle(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			hdl = c_void_p()
			self.assertCall( lib.sp_make_handle(byref(hdl)) )
			self.assertCall( lib.sp_destroy_handle(hdl) )

	def test_version(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			major = c_int()
			minor = c_int()
			change = c_int()
			status = c_int()

			lib.optkit_version(byref(major), byref(minor), byref(change),
								  byref(status))

			version = self.version_string(major.value, minor.value,
										 change.value, status.value)

			self.assertNotEqual( version, '0.0.0' )
			if self.VERBOSE_TEST:
				print("sparselib version", sversion)

class SparseMatrixTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = SparseLinsysLibs()
		self.A_test_sparse = self.A_test_sparse_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_allocate(self):
		shape = (m, n) = self.shape
		nnz = int(0.05 * m * n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, 101)

			# rowmajor: calloc, free
			self.assertCall( lib.sp_matrix_calloc(A, m, n, nnz,
												  lib.enums.CblasRowMajor) )
			self.register_var('A', A, lib.sp_matrix_free)

			self.assertEqual( A.size1, m )
			self.assertEqual( A.size2, n )
			self.assertEqual( A.nnz, nnz )
			self.assertEqual( A.ptrlen, m + 1 )
			self.assertEqual( A.order, lib.enums.CblasRowMajor )
			if not gpu:
				map(lambda i : self.assertEqual(A.val[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ind[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ptr[i], 0),
					xrange(2 + m + n))

			self.assertCall( lib.sp_matrix_free(A) )
			self.assertEqual( A.size2, 0 )
			self.assertEqual( A.nnz, 0 )
			self.assertEqual( A.ptrlen, 0 )

			# colmajor: calloc, free
			self.assertCall( lib.sp_matrix_calloc(A, m, n, nnz,
												  lib.enums.CblasColMajor) )
			self.assertEqual( A.size1, m )
			self.assertEqual( A.size2, n )
			self.assertEqual( A.nnz, nnz )
			self.assertEqual( A.ptrlen, n + 1 )
			self.assertEqual( A.order, lib.enums.CblasColMajor )
			if not gpu:
				map(lambda i : self.assertEqual(A.val[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ind[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ptr[i], 0),
					xrange(2 + m + n))

			self.free_var('A')
			self.assertEqual( A.size1, 0 )
			self.assertEqual( A.size2, 0 )
			self.assertEqual( A.nnz, 0 )
			self.assertEqual( A.ptrlen, 0 )

			self.assertCall( lib.ok_device_reset() )

	def test_io(self):
		shape = (m, n) = self.shape
		x_rand = np.random.rand(n)
		A_test = self.A_test_sparse
		B_test = A_test * (1 + np.random.rand(m, n))

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				# sparse handle
				hdl = self.register_sparse_handle(lib, 'hdl')

				A, A_, A_py, A_val, A_ind, A_ptr = self.register_sparsemat(
						lib, A_test, order, 'A')
				B, B_, B_py, B_val, B_ind, B_ptr = self.register_sparsemat(
						lib, B_test, order, 'B')

				# Ax == A_py * x (initalized to rand)
				Ax_py = A_.dot(x_rand)
				Ax_c = A_py * x_rand
				self.assertVecEqual( Ax_py, Ax_c, ATOLM, RTOL )

				# Test sparse copy optkit->python
				# A_ * x != A_c * x (calloc to zero)
				self.assertCall( lib.sp_matrix_memcpy_am(A_val, A_ind, A_ptr,
														 A) )
				Ax_py = A_.dot(x_rand)
				Ax_c = A_py * x_rand
				self.assertVecNotEqual( Ax_py, Ax_c, ATOLM, RTOL )
				self.assertVecEqual( 0, Ax_c, ATOLM, 0)

				# Test sparse copy python->optkit
				# B_py -> B_c -> B_py
				# B_py * x == B_ * x
				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, B, B_val, B_ind,
														 B_ptr) )
				self.assertCall( lib.sp_matrix_memcpy_am(B_val, B_ind, B_ptr,
														 B) )
				Bx_py = B_.dot(x_rand)
				Bx_c = B_py * x_rand
				self.assertVecEqual( Bx_py, Bx_c, ATOLM, RTOL )

				# Test sparse copy optkit->optkit
				# B_c -> A_c -> A_py
				# B_py * x == A_py * x
				self.assertCall( lib.sp_matrix_memcpy_mm(A, B) )
				self.assertCall( lib.sp_matrix_memcpy_am(A_val, A_ind, A_ptr,
														 A) )
				Ax = A_py * x_rand
				Bx = B_py * x_rand
				self.assertVecEqual( Ax, Bx, ATOLM, RTOL )

				# Test sparse value copy optkit->python
				# A_py *= 0
				# A_c -> A_py (values)
				# B_py * x == A_py * x (still)
				A_py *= 0
				self.assertCall( lib.sp_matrix_memcpy_vals_am(A_val, A) )
				Ax_c = A_py * x_rand
				Bx_c = B_py * x_rand
				self.assertVecEqual( Ax_c, Bx_c, ATOLM, RTOL )

				# Test sparse value copy python->optkit
				# A_py *= 2; A_py -> A_c; A_py *= 0; A_c -> A_py
				# 2 * B_py * x == A_py * x
				A_py *= 2
				self.assertCall( lib.sp_matrix_memcpy_vals_ma(hdl, A, A_val) )
				A_py *= 0
				self.assertCall( lib.sp_matrix_memcpy_vals_am(A_val, A) )
				Ax_c = A_py * x_rand
				Bx_c = B_py * x_rand
				self.assertVecEqual( Ax_c, 2 * Bx_c, ATOLM, RTOL )

				# Test sparse value copy optkit->optkit
				# A_c -> B_c -> B_py
				# B_py * x == A_py * x
				self.assertCall( lib.sp_matrix_memcpy_vals_mm(B, A) )
				self.assertCall( lib.sp_matrix_memcpy_vals_am(B_val, B) )
				Ax_c = A_py * x_rand
				Bx_c = B_py * x_rand
				self.assertVecEqual( Ax_c, Bx_c, ATOLM, RTOL )

				self.free_vars('A', 'B', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_multiply(self):
		shape = (m, n) = self.shape
		x_rand = np.random.rand(n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				# sparse handle
				hdl = self.register_sparse_handle(lib, 'hdl')

				A, A_, A_py, A_val, A_ind, A_ptr = self.register_sparsemat(
						lib, self.A_test_sparse, order, 'A')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')

				x_py[:] = x_rand[:]

				# load A_py -> A_c
				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val,
														 A_ind, A_ptr) )

				# y = Ax, Py vs. C
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				Ax = A_py * x_rand
				self.assertVecEqual( Ax, y_py, ATOLM, RTOL )

				# x = A'y, Py vs. C
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasTrans, 1,
												  A, y, 0, x) )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				Ay = A_py.T * Ax
				self.assertVecEqual( Ay, x_py, ATOLN, RTOL )

				# y = alpha Ax + beta y, Py vs. C
				alpha = np.random.rand()
				beta = np.random.rand()

				result = alpha * A_py * x_py + beta * y_py
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  alpha, A, x, beta, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertVecEqual( result, y_py, ATOLM, RTOL )

				self.free_vars('x', 'y', 'A', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_elementwise_transformations(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				# sparse handle
				hdl = self.register_sparse_handle(lib, 'hdl')

				A, A_, A_py, A_val, A_ind, A_ptr = self.register_sparsemat(
						lib, self.A_test_sparse, order, 'A')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')
				self.register_var('A', A, lib.sp_matrix_free)
				self.register_var('x', x, lib.vector_free)
				self.register_var('y', y, lib.vector_free)

				amax = A_py.data.max()
				x_rand = np.random.rand(n)
				x_py[:] = x_rand[:]
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

				# abs
				# make A_py mixed sign, load to A_c. then, A = abs(A)
				A_py.data -= (amax / 2.)
				A_ = np.abs(A_py.toarray())

				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val,
														 A_ind, A_ptr) )
				self.assertCall( lib.sp_matrix_abs(A) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertVecEqual( A_.dot(x_rand), y_py, ATOLM, RTOL )

				# pow
				# A is nonnegative from previous step. set A_ij = A_ij ^ p
				p = 3 * np.random.rand()
				A_ **= p
				self.assertCall( lib.sp_matrix_pow(A, p) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertVecEqual( A_.dot(x_rand), y_py, ATOLM, RTOL )

				# scale
				# A = alpha * A
				alpha = -1 + 2 * np.random.rand()
				A_ *= alpha
				self.assertCall( lib.sp_matrix_scale(A, alpha) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertVecEqual( A_.dot(x_rand), y_py, ATOLM, RTOL )

				self.free_var('x')
				self.free_var('y')
				self.free_var('A')
				self.free_var('hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_diagonal_scaling(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				# sparse handle
				hdl = self.register_sparse_handle(lib, 'hdl')

				A, A_, A_py, A_val, A_ind, A_ptr = self.register_sparsemat(
						lib, self.A_test_sparse, order, 'A')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')
				d, d_py, d_ptr = self.register_vector(lib, m, 'd')
				e, e_py, e_ptr = self.register_vector(lib, n, 'e')

				amax = A_py.data.max()
				x_rand = np.random.rand(n)
				x_py[:] = x_rand[:]
				d_py[:] = np.random.rand(m)
				e_py[:] = np.random.rand(n)

				# load x_py -> x, A_py -> A, d_py -> d, e_py -> e
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val,
														 A_ind, A_ptr) )
				self.assertCall( lib.vector_memcpy_va(d, d_ptr, 1) )
				self.assertCall( lib.vector_memcpy_va(e, e_ptr, 1) )

				# scale_left: A = diag(d) * A
				# (form diag (d) * A * x, compare Py vs. C)
				self.assertCall( lib.sp_matrix_scale_left(hdl, A, d) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				result = (np.diag(d_py) * A_py).dot(x_py)
				self.assertVecEqual( result, y_py, ATOLM, RTOL )

				# scale_right: A = A * diag(e)
				# (form diag (d) * A * diag(e) * x, compare Py vs. C)
				self.assertCall( lib.sp_matrix_scale_right(hdl, A, e) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				result = (np.diag(d_py) * A_py).dot(e_py * x_py)
				self.assertVecEqual( result, y_py, ATOLM, RTOL )

				self.free_vars('x', 'y', 'd', 'e', 'A', 'hdl')
				self.assertEqual(lib.ok_device_reset(), 0)
