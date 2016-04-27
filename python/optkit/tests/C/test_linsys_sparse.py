import os
import numpy as np
import scipy.sparse as sp
from ctypes import c_int, c_uint, Structure, byref, c_void_p
from optkit.libs import SparseLinsysLibs
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
		self.assertTrue(any(libs))

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

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()

	@staticmethod
	def make_vec_triplet(lib, size_):
		a = lib.vector(0, 0, None)
		lib.vector_calloc(a, size_)
		a_py = np.zeros(size_).astype(lib.pyfloat)
		a_ptr = a_py.ctypes.data_as(lib.ok_float_p)
		return a, a_py, a_ptr

	@staticmethod
	def make_spmat_quintet(lib, shape, rowmajor=True):
		fmt = 'csr' if rowmajor else 'csc'
		order = 101 if rowmajor else 102
		sp_constructor = sp.csr_matrix if rowmajor else sp.csc_matrix

		A_py = sp.rand(shape[0], shape[1], density=0.1, format=fmt,
					   dtype=lib.pyfloat)

		A_ptr_p = A_py.indptr.ctypes.data_as(lib.ok_int_p)
		A_ind_p = A_py.indices.ctypes.data_as(lib.ok_int_p)
		A_val_p = A_py.data.ctypes.data_as(lib.ok_float_p)
		A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
		lib.sp_matrix_calloc(A, shape[0], shape[1], A_py.nnz, order)
		return A, A_py, A_ptr_p, A_ind_p, A_val_p

	def test_allocate(self):
		shape = (m, n) = self.shape
		nnz = int(0.05 * m * n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

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

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for rowmajor in (True, False):
				# sparse handle
				hdl = c_void_p()
				self.assertCall( lib.sp_make_handle(byref(hdl)) )
				self.register_var('hdl', hdl, lib.sp_destroy_handle)

				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				B, B_py, B_ptr_p, B_ind_p, B_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				self.register_var('A', A, lib.sp_matrix_free)
				self.register_var('B', B, lib.sp_matrix_free)

				x_rand = np.random.rand(n)
				A_copy = A_py.toarray()
				B_copy = B_py.toarray()

				# Acopy * x == A_py * x (initalized to rand)
				Ax_py = A_copy.dot(x_rand)
				Ax_c = A_py * x_rand
				self.assertTrue( np.linalg.norm(Ax_py - Ax_c) <=
								 ATOLM + RTOL * np.linalg.norm(Ax_py) )

				# Test sparse copy optkit->python
				# Acopy * x != A_c * x (calloc to zero)
				self.assertCall( lib.sp_matrix_memcpy_am(A_val_p, A_ind_p,
														 A_ptr_p, A) )
				Ax_py = A_copy.dot(x_rand)
				Ax_c = A_py * x_rand
				self.assertFalse( np.linalg.norm(Ax_py - Ax_c) <=
								 ATOLM + RTOL * np.linalg.norm(Ax_py) )
				self.assertTrue( np.linalg.norm(Ax_c) <= ATOLM )

				# Test sparse copy python->optkit
				# B_py -> B_c -> B_py
				# B_py * x == B_copy * x
				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, B, B_val_p,
												 		 B_ind_p, B_ptr_p) )
				self.assertCall( lib.sp_matrix_memcpy_am(B_val_p, B_ind_p,
														 B_ptr_p, B) )
				Bx_py = B_copy.dot(x_rand)
				Bx_c = B_py * x_rand
				self.assertTrue( np.linalg.norm(Bx_py - Bx_c) <=
								 ATOLM + RTOL * np.linalg.norm(Bx_py) )

				# Test sparse copy optkit->optkit
				# B_c -> A_c -> A_py
				# B_py * x == A_py * x
				self.assertCall( lib.sp_matrix_memcpy_mm(A, B) )
				self.assertCall( lib.sp_matrix_memcpy_am(A_val_p, A_ind_p,
														 A_ptr_p, A) )
				Ax_c = A_py * x_rand
				Bx_c = B_py * x_rand
				self.assertTrue( np.linalg.norm(Ax_c - Bx_c) <=
								 ATOLM + RTOL * np.linalg.norm(Bx_c) )

				# Test sparse value copy optkit->python
				# A_py *= 0
				# A_c -> A_py (values)
				# B_py * x == A_py * x (still)
				A_py *= 0
				self.assertCall( lib.sp_matrix_memcpy_vals_am(A_val_p, A) )
				Ax_c = A_py * x_rand
				Bx_c = B_py * x_rand
				self.assertTrue( np.linalg.norm(Ax_c - Bx_c) <=
								 ATOLM + RTOL * np.linalg.norm(Bx_c) )


				# Test sparse value copy python->optkit
				# A_py *= 2; A_py -> A_c; A_py *= 0; A_c -> A_py
				# 2 * B_py * x == A_py * x
				A_py *= 2
				self.assertCall( lib.sp_matrix_memcpy_vals_ma(hdl, A,
															  A_val_p) )
				A_py *= 0
				self.assertCall( lib.sp_matrix_memcpy_vals_am(A_val_p, A) )
				Ax_c = A_py * x_rand
				Bx_c = B_py * x_rand
				self.assertTrue( np.linalg.norm(Ax_c - 2 * Bx_c) <=
								 ATOLM + RTOL * 2 * np.linalg.norm(Bx_c) )

				# Test sparse value copy optkit->optkit
				# A_c -> B_c -> B_py
				# B_py * x == A_py * x
				self.assertCall( lib.sp_matrix_memcpy_vals_mm(B, A) )
				self.assertCall( lib.sp_matrix_memcpy_vals_am(B_val_p, B) )
				Ax_c = A_py * x_rand
				Bx_c = B_py * x_rand
				self.assertTrue( np.linalg.norm(Ax_c - Bx_c) <=
								 ATOLM + RTOL * np.linalg.norm(Bx_c) )

				self.free_var('A')
				self.free_var('B')
				self.free_var('hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_multiply(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			for rowmajor in (True, False):
				# sparse handle
				hdl = c_void_p()
				self.assertCall( lib.sp_make_handle(byref(hdl)) )
				self.register_var('hdl', hdl, lib.sp_destroy_handle)

				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(lib, n)
				y, y_py, y_ptr = self.make_vec_triplet(lib, m)
				self.register_var('A', A, lib.sp_matrix_free)
				self.register_var('x', x, lib.vector_free)
				self.register_var('y', y, lib.vector_free)

				x_rand = np.random.rand(n)
				x_py[:] = x_rand[:]

				# load A_py -> A_c
				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val_p,
														 A_ind_p, A_ptr_p) )

				# y = Ax, Py vs. C
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				Ax = A_py * x_rand
				self.assertTrue( np.linalg.norm(Ax - y_py) <=
								 ATOLM + RTOL * np.linalg.norm(y_py) )

				# x = A'y, Py vs. C
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasTrans, 1,
												  A, y, 0, x) )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				Ay = A_py.T * Ax
				self.assertTrue( np.linalg.norm(Ay - x_py) <=
								 ATOLN + RTOL * np.linalg.norm(x_py) )

				# y = alpha Ax + beta y, Py vs. C
				alpha = np.random.rand()
				beta = np.random.rand()

				result = alpha * A_py * x_py + beta * y_py
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  alpha, A, x, beta, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertTrue( np.linalg.norm(result - y_py) <=
								 ATOLM + RTOL * np.linalg.norm(result) )

				self.free_var('x')
				self.free_var('y')
				self.free_var('A')
				self.free_var('hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_elementwise_transformations(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for rowmajor in (True, False):
				# sparse handle
				hdl = c_void_p()
				self.assertCall( lib.sp_make_handle(byref(hdl)) )
				self.register_var('hdl', hdl, lib.sp_destroy_handle)

				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(lib, n)
				y, y_py, y_ptr = self.make_vec_triplet(lib, m)
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
				A_copy = np.abs(A_py.toarray())

				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val_p,
														 A_ind_p, A_ptr_p) )
				self.assertCall( lib.sp_matrix_abs(A) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertTrue( np.linalg.norm(A_copy.dot(x_rand) - y_py) <=
								 ATOLM + RTOL * np.linalg.norm(y_py) )

				# pow
				# A is nonnegative from previous step. set A_ij = A_ij ^ p
				p = 3 * np.random.rand()
				A_copy **= p
				self.assertCall( lib.sp_matrix_pow(A, p) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertTrue( np.linalg.norm(A_copy.dot(x_rand) - y_py) <=
								 ATOLM + RTOL * np.linalg.norm(y_py) )

				# scale
				# A = alpha * A
				alpha = -1 + 2 * np.random.rand()
				A_copy *= alpha
				self.assertCall( lib.sp_matrix_scale(A, alpha) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertTrue( np.linalg.norm(A_copy.dot(x_rand) - y_py) <=
								 ATOLM + RTOL * np.linalg.norm(y_py) )

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

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for rowmajor in (True, False):
				# sparse handle
				hdl = c_void_p()
				self.assertCall( lib.sp_make_handle(byref(hdl)) )
				self.register_var('hdl', hdl, lib.sp_destroy_handle)

				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(lib, n)
				y, y_py, y_ptr = self.make_vec_triplet(lib, m)
				d, d_py, d_ptr = self.make_vec_triplet(lib, m)
				e, e_py, e_ptr = self.make_vec_triplet(lib, n)
				self.register_var('A', A, lib.sp_matrix_free)
				self.register_var('x', x, lib.vector_free)
				self.register_var('y', y, lib.vector_free)
				self.register_var('d', d, lib.vector_free)
				self.register_var('e', e, lib.vector_free)

				amax = A_py.data.max()
				x_rand = np.random.rand(n)
				x_py[:] = x_rand[:]
				d_py[:] = np.random.rand(m)
				e_py[:] = np.random.rand(n)

				# load x_py -> x, A_py -> A, d_py -> d, e_py -> e
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( lib.sp_matrix_memcpy_ma(hdl, A, A_val_p,
														 A_ind_p, A_ptr_p) )
				self.assertCall( lib.vector_memcpy_va(d, d_ptr, 1) )
				self.assertCall( lib.vector_memcpy_va(e, e_ptr, 1) )

				# scale_left: A = diag(d) * A
				# (form diag (d) * A * x, compare Py vs. C)
				self.assertCall( lib.sp_matrix_scale_left(hdl, A, d) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				result = (np.diag(d_py) * A_py).dot(x_py)
				self.assertTrue( np.linalg.norm(result - y_py) <=
								 ATOLM + RTOL * np.linalg.norm(result) )

				# scale_right: A = A * diag(e)
				# (form diag (d) * A * diag(e) * x, compare Py vs. C)
				self.assertCall( lib.sp_matrix_scale_right(hdl, A, e) )
				self.assertCall( lib.sp_blas_gemv(hdl, lib.enums.CblasNoTrans,
												  1, A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				result = (np.diag(d_py) * A_py).dot(e_py * x_py)
				self.assertTrue( np.linalg.norm(result - y_py) <=
								 ATOLM + RTOL * np.linalg.norm(result) )

				self.free_var('x')
				self.free_var('y')
				self.free_var('d')
				self.free_var('e')
				self.free_var('A')
				self.free_var('hdl')
				self.assertEqual(lib.ok_device_reset(), 0)