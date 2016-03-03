import unittest
import os
import numpy as np
import scipy.sparse as sp
from ctypes import c_int, c_uint, Structure, byref, c_void_p
from optkit.libs import DenseLinsysLibs, SparseLinsysLibs
from optkit.tests.defs import VERBOSE_TEST, CONDITIONS, version_string, \
							  DEFAULT_SHAPE

class SparseLibsTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.sparse_libs = SparseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		dlibs = []
		slibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(single_precision=single_precision,
											 gpu=gpu))
			slibs.append(self.sparse_libs.get(
						 		dlibs[-1], single_precision=single_precision,
						 		gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(slibs))

	def test_lib_types(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			self.assertTrue('ok_int_p' in dir(lib))
			self.assertTrue('ok_float_p' in dir(lib))
			self.assertTrue('sparse_matrix' in dir(lib))
			self.assertTrue('sparse_matrix_p' in dir(lib))

	def test_sparse_handle(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			hdl = c_void_p()
			self.assertEqual(lib.sp_make_handle(byref(hdl)), 0)
			self.assertEqual(lib.sp_destroy_handle(byref(hdl)), 0)

	def test_version(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			major = c_int()
			minor = c_int()
			change = c_int()
			status = c_int()

			dlib.denselib_version(byref(major), byref(minor), byref(change),
								  byref(status))

			dversion = version_string(major.value, minor.value, change.value,
									  status.value)

			self.assertNotEqual(dversion, '0.0.0')

			lib.sparselib_version(byref(major), byref(minor), byref(change),
								  byref(status))

			sversion = version_string(major.value, minor.value, change.value,
									  status.value)

			self.assertNotEqual(sversion, '0.0.0')
			self.assertEqual(sversion, dversion)
			if VERBOSE_TEST:
				print("sparselib version", sversion)

class SparseMatrixTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.sparse_libs = SparseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = DEFAULT_SHAPE


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

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, 101)

			# rowmajor: calloc, free
			lib.sp_matrix_calloc(A, m, n, nnz, dlib.enums.CblasRowMajor)
			self.assertEqual(A.size1, m)
			self.assertEqual(A.size2, n)
			self.assertEqual(A.nnz, nnz)
			self.assertEqual(A.ptrlen, m + 1)
			self.assertEqual(A.order, 101)
			if not gpu:
				map(lambda i : self.assertEqual(A.val[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ind[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ptr[i], 0),
					xrange(2 + m + n))

			lib.sp_matrix_free(A)
			self.assertEqual(A.size2, 0)
			self.assertEqual(A.nnz, 0)
			self.assertEqual(A.ptrlen, 0)

			# colmajor: calloc, free
			lib.sp_matrix_calloc(A, m, n, nnz, dlib.enums.CblasColMajor)
			self.assertEqual(A.size1, m)
			self.assertEqual(A.size2, n)
			self.assertEqual(A.nnz, nnz)
			self.assertEqual(A.ptrlen, n + 1)
			self.assertEqual(A.order, 102)
			if not gpu:
				map(lambda i : self.assertEqual(A.val[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ind[i], 0), xrange(2 * nnz))
				map(lambda i : self.assertEqual(A.ptr[i], 0),
					xrange(2 + m + n))

			lib.sp_matrix_free(A)
			self.assertEqual(A.size1, 0)
			self.assertEqual(A.size2, 0)
			self.assertEqual(A.nnz, 0)
			self.assertEqual(A.ptrlen, 0)

			self.assertEqual(dlib.ok_device_reset(), 0)


	def test_io(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			# sparse handle
			hdl = c_void_p()
			self.assertEqual(lib.sp_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):

				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				B, B_py, B_ptr_p, B_ind_p, B_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)

				x_rand = np.random.rand(n)
				A_copy = A_py.toarray()
				B_copy = B_py.toarray()


				# Acopy * x == A_py * x (initalized to rand)
				self.assertTrue(np.allclose(A_copy.dot(x_rand), A_py * x_rand,
											5))

				# Test sparse copy optkit->python
				# Acopy * x != A_c * x (calloc to zero)
				lib.sp_matrix_memcpy_am(A_val_p, A_ind_p, A_ptr_p, A)
				self.assertFalse(np.allclose(A_copy.dot(x_rand), A_py * x_rand,
											 5))
				self.assertTrue(all(map(lambda x : x == 0, A_py * x_rand)))

				# Test sparse copy python->optkit
				# B_py -> B_c -> B_py
				# B_py * x == B_copy * x
				lib.sp_matrix_memcpy_ma(hdl, B, B_val_p, B_ind_p, B_ptr_p)
				lib.sp_matrix_memcpy_am(B_val_p, B_ind_p, B_ptr_p, B)
				self.assertTrue(np.allclose(B_copy.dot(x_rand), B_py * x_rand))

				# Test sparse copy optkit->optkit
				# B_c -> A_c -> A_py
				# B_py * x == A_py * x
				lib.sp_matrix_memcpy_mm(A, B)
				lib.sp_matrix_memcpy_am(A_val_p, A_ind_p, A_ptr_p, A)
				self.assertTrue(np.allclose(A_py * x_rand, B_py * x_rand))

				# Test sparse value copy optkit->python
				# A_py *= 0
				# A_c -> A_py (values)
				# B_py * x == A_py * x (still)
				A_py *= 0
				lib.sp_matrix_memcpy_vals_am(A_val_p, A)
				self.assertTrue(np.allclose(A_py * x_rand, B_py * x_rand))

				# Test sparse value copy python->optkit
				# A_py *= 2; A_py -> A_c; A_py *= 0; A_c -> A_py
				# 2 * B_py * x == A_py * x
				A_py *= 2
				lib.sp_matrix_memcpy_vals_ma(hdl, A, A_val_p)
				A_py *= 0
				lib.sp_matrix_memcpy_vals_am(A_val_p, A)
				self.assertTrue(np.allclose(A_py * x_rand, 2 * B_py * x_rand))

				# Test sparse value copy optkit->optkit
				# A_c -> B_c -> B_py
				# B_py * x == A_py * x
				lib.sp_matrix_memcpy_vals_mm(B, A)
				lib.sp_matrix_memcpy_vals_am(B_val_p, B)
				self.assertTrue(np.allclose(A_py * x_rand, B_py * x_rand))

			self.assertEqual(lib.sp_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_multiply(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			# sparse handle
			hdl = c_void_p()
			self.assertEqual(lib.sp_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(dlib, n)
				y, y_py, y_ptr = self.make_vec_triplet(dlib, m)

				x_rand = np.random.rand(n)
				x_py[:] = x_rand[:]

				# load A_py -> A_c
				lib.sp_matrix_memcpy_ma(hdl, A, A_val_p, A_ind_p, A_ptr_p)

				# y = Ax, Py vs. C
				dlib.vector_memcpy_va(x, x_ptr, 1)
				lib.sp_blas_gemv(hdl, dlib.enums.CblasNoTrans, 1, A, x, 0, y)
				dlib.vector_memcpy_av(y_ptr, y, 1)
				self.assertTrue(np.allclose(A_py * x_rand, y_py))

				# x = A'y, Py vs. C
				lib.sp_blas_gemv(hdl, dlib.enums.CblasTrans, 1, A, y, 0, x)
				dlib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.allclose(A_py.T * A_py * x_rand, x_py))

				# y = alpha Ax + beta y, Py vs. C
				alpha = np.random.rand()
				beta = np.random.rand()

				result = alpha * A_py * x_py + beta * y_py
				lib.sp_blas_gemv(hdl, dlib.enums.CblasNoTrans, alpha, A, x,
								 beta, y)
				dlib.vector_memcpy_av(y_ptr, y, 1)
				self.assertTrue(np.allclose(result, y_py))

			self.assertEqual(lib.sp_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_elementwise_transformations(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			# sparse handle
			hdl = c_void_p()
			self.assertEqual(lib.sp_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):

				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(dlib, n)
				y, y_py, y_ptr = self.make_vec_triplet(dlib, m)

				amax = A_py.data.max()
				x_rand = np.random.rand(n)
				x_py[:] = x_rand[:]
				dlib.vector_memcpy_va(x, x_ptr, 1)

				# abs
				# make A_py mixed sign, load to A_c. then, A = abs(A)
				A_py.data -= (amax / 2.)
				A_copy = np.abs(A_py.toarray())

				lib.sp_matrix_memcpy_ma(hdl, A, A_val_p, A_ind_p, A_ptr_p)
				lib.sp_matrix_abs(A)
				lib.sp_blas_gemv(hdl, dlib.enums.CblasNoTrans, 1, A, x, 0, y)
				dlib.vector_memcpy_av(y_ptr, y, 1)
				self.assertTrue(np.allclose(y_py, A_copy.dot(x_rand)))

				# pow
				# A is nonnegative from previous step. set A_ij = A_ij ^ p
				p = 3 * np.random.rand()
				A_copy **= p
				lib.sp_matrix_pow(A, p)
				lib.sp_blas_gemv(hdl, dlib.enums.CblasNoTrans, 1, A, x, 0, y)
				dlib.vector_memcpy_av(y_ptr, y, 1)
				self.assertTrue(np.allclose(y_py, A_copy.dot(x_rand)))

				# scale
				# A = alpha * A
				alpha = -1 + 2 * np.random.rand()
				A_copy *= alpha
				lib.sp_matrix_scale(A, alpha)
				lib.sp_blas_gemv(hdl, dlib.enums.CblasNoTrans, 1, A, x, 0, y)
				dlib.vector_memcpy_av(y_ptr, y, 1)
				self.assertTrue(np.allclose(y_py, A_copy.dot(x_rand)))

			self.assertEqual(lib.sp_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_diagonal_scaling(self):
		shape = (m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.sparse_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			if lib is None:
				continue

			# sparse handle
			hdl = c_void_p()
			self.assertEqual(lib.sp_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):

				A, A_py, A_ptr_p, A_ind_p, A_val_p = self.make_spmat_quintet(
						lib, shape, rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(dlib, n)
				y, y_py, y_ptr = self.make_vec_triplet(dlib, m)
				d, d_py, d_ptr = self.make_vec_triplet(dlib, m)
				e, e_py, e_ptr = self.make_vec_triplet(dlib, n)

				amax = A_py.data.max()
				x_rand = np.random.rand(n)
				x_py[:] = x_rand[:]
				d_py[:] = np.random.rand(m)
				e_py[:] = np.random.rand(n)

				# load x_py -> x, A_py -> A, d_py -> d, e_py -> e
				dlib.vector_memcpy_va(x, x_ptr, 1)
				lib.sp_matrix_memcpy_ma(hdl, A, A_val_p, A_ind_p, A_ptr_p)
				dlib.vector_memcpy_va(d, d_ptr, 1)
				dlib.vector_memcpy_va(e, e_ptr, 1)

				# scale_left: A = diag(d) * A
				# (form diag (d) * A * x, compare Py vs. C)
				lib.sp_matrix_scale_left(hdl, A, d)
				lib.sp_blas_gemv(hdl, dlib.enums.CblasNoTrans, 1, A, x, 0, y)
				dlib.vector_memcpy_av(y_ptr, y, 1)
				result = (np.diag(d_py) * A_py).dot(x_py)
				self.assertTrue(np.allclose(y_py, result))

				# scale_right: A = A * diag(e)
				# (form diag (d) * A * diag(e) * x, compare Py vs. C)
				lib.sp_matrix_scale_right(hdl, A, e)
				lib.sp_blas_gemv(hdl, dlib.enums.CblasNoTrans, 1, A, x, 0, y)
				dlib.vector_memcpy_av(y_ptr, y, 1)
				result = (np.diag(d_py) * A_py).dot(e_py * x_py)
				self.assertTrue(np.allclose(y_py,  result))

			self.assertEqual(lib.sp_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)