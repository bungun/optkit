import unittest
import os
import numpy as np
from ctypes import c_float, c_int, c_size_t, c_void_p, Structure, byref
from optkit.libs import DenseLinsysLibs, SparseLinsysLibs
from optkit.tests.defs import VERBOSE_TEST, CONDITIONS, version_string, \
							  DEFAULT_SHAPE, DEFAULT_MATRIX_PATH, \
							  significant_digits

class DenseLibsTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in CONDITIONS:
			libs.append(self.dense_libs.get(
					single_precision=single_precision, gpu=gpu))
		self.assertTrue(any(libs))

	def test_lib_types(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertTrue('ok_float' in dir(lib))
			self.assertTrue('ok_int' in dir(lib))
			self.assertTrue('c_int_p' in dir(lib))
			self.assertTrue('ok_float_p' in dir(lib))
			self.assertTrue('ok_int_p' in dir(lib))
			self.assertTrue('vector' in dir(lib))
			self.assertTrue('vector_p' in dir(lib))
			self.assertTrue('matrix' in dir(lib))
			self.assertTrue('matrix_p' in dir(lib))
			self.assertTrue(single_precision == (lib.ok_float == c_float))
			self.assertTrue(lib.ok_int == c_int)

	def test_blas_handle(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			handle = c_void_p()
			# create
			self.assertEqual(lib.blas_make_handle(byref(handle)), 0)
			# destroy
			self.assertEqual(lib.blas_destroy_handle(handle), 0)

	def test_device_reset(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			# reset
			self.assertEqual(lib.ok_device_reset(), 0)

			# allocate - deallocate - reset
			handle = c_void_p()
			self.assertEqual(lib.blas_make_handle(byref(handle)), 0)
			self.assertEqual(lib.blas_destroy_handle(handle), 0)
			self.assertEqual(lib.ok_device_reset(), 0)


	def test_version(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			major = c_int()
			minor = c_int()
			change = c_int()
			status = c_int()

			lib.denselib_version(byref(major), byref(minor), byref(change),
								 byref(status))

			version = version_string(major.value, minor.value, change.value,
									 status.value)

			self.assertNotEqual(version, '0.0.0')
			if VERBOSE_TEST:
				print("denselib version", version)

class DenseBLASTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)

	@staticmethod
	def make_vec_triplet(lib, size_):
			a = lib.vector(0, 0, None)
			lib.vector_calloc(a, size_)
			a_py = np.zeros(size_).astype(lib.pyfloat)
			a_ptr = a_py.ctypes.data_as(lib.ok_float_p)
			return a, a_py, a_ptr

	@staticmethod
	def make_mat_triplet(lib, shape, rowmajor=True):
		order = 101 if rowmajor else 102
		pyorder = 'C' if rowmajor else 'F'
		A = lib.matrix(0, 0, 0, None, order)
		lib.matrix_calloc(A, shape[0], shape[1], order)
		A_py = np.zeros(shape, order=pyorder).astype(lib.pyfloat)
		A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
		return A, A_py, A_ptr

	def test_blas1_dot(self):
		(m, n) = self.shape
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			v, v_py, v_ptr = self.make_vec_triplet(lib, m)
			w, w_py, w_ptr = self.make_vec_triplet(lib, m)

			v_py += np.random.rand(m)
			w_py += np.random.rand(m)
			lib.vector_memcpy_va(v, v_ptr, 1)
			lib.vector_memcpy_va(w, w_ptr, 1)
			answer = lib.blas_dot(hdl, v, w)

			self.assertAlmostEqual(significant_digits(answer),
								   significant_digits(v_py.dot(w_py)), DIGITS)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_blas1_nrm2(self):
		(m, n) = self.shape
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			v, v_py, v_ptr = self.make_vec_triplet(lib, m)
			v_py += np.random.rand(m)
			lib.vector_memcpy_va(v, v_ptr, 1)
			answer = lib.blas_nrm2(hdl, v)
			self.assertAlmostEqual(significant_digits(answer),
								   significant_digits(np.linalg.norm(v_py)),
								   DIGITS)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_blas1_asum(self):
		(m, n) = self.shape
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			v, v_py, v_ptr = self.make_vec_triplet(lib, m)
			v_py += np.random.rand(m)
			lib.vector_memcpy_va(v, v_ptr, 1)
			answer = lib.blas_asum(hdl, v)
			self.assertAlmostEqual(significant_digits(answer),
								   significant_digits(np.linalg.norm(v_py, 1)),
								   DIGITS)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)


	def test_blas1_scal(self):
		(m, n) = self.shape
		hdl = c_void_p()
		v_rand= np.random.rand(m)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			v, v_py, v_ptr = self.make_vec_triplet(lib, m)
			v_py += v_rand
			lib.vector_memcpy_va(v, v_ptr, 1)
			alpha = np.random.rand()
			lib.blas_scal(hdl, alpha, v)
			lib.vector_memcpy_av(v_ptr, v, 1)
			self.assertTrue(np.allclose(v_py, alpha * v_rand, DIGITS))

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)


	def test_blas1_axpy(self):
		(m, n) = self.shape
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			v, v_py, v_ptr = self.make_vec_triplet(lib, m)
			w, w_py, w_ptr = self.make_vec_triplet(lib, m)

			v_py += np.random.rand(m)
			w_py += np.random.rand(m)
			alpha = np.random.rand()
			pyresult = alpha * v_py + w_py
			lib.vector_memcpy_va(v, v_ptr, 1)
			lib.vector_memcpy_va(w, w_ptr, 1)
			lib.blas_axpy(hdl, alpha, v, w)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.allclose(w_py, pyresult, DIGITS))

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_blas2_gemv(self):
		(m, n) = self.shape
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				# make A, x, y
				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n),
													   rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(lib, n)
				y, y_py, y_ptr = self.make_vec_triplet(lib, m)

				# populate A, x, y (in Py and C)
				A_py += self.A_test
				x_py += np.random.rand(n)
				y_py += np.random.rand(m)
				lib.vector_memcpy_va(x, x_ptr, 1)
				lib.vector_memcpy_va(y, y_ptr, 1)
				lib.matrix_memcpy_ma(A, A_ptr, order)

				# perform y = alpha * A * x + beta *  y
				alpha = -0.5 + np.random.rand()
				beta = -0.5 + np.random.rand()

				pyresult = alpha * A_py.dot(x_py) + beta * y_py
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, alpha, A, x, beta,
							  y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				self.assertTrue(np.allclose(y_py, pyresult, DIGITS))

				# perform x = alpha * A' * y + beta * x
				y_py[:] = pyresult[:]
				pyresult = alpha * A_py.T.dot(y_py) + beta * x_py
				lib.blas_gemv(hdl, lib.enums.CblasTrans, alpha, A, y, beta, x)
				lib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.allclose(x_py, pyresult, DIGITS))

				lib.matrix_free(A)
				lib.vector_free(x)
				lib.vector_free(y)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_blas2_trsv(self):
		(m, n) = self.shape
		hdl = c_void_p()

		# generate lower triangular matrix L
		L_test = self.A_test.T.dot(self.A_test)

		# normalize L so inversion doesn't blow up
		L_test /= np.linalg.norm(L_test)

		for i in xrange(n):
			# diagonal entries ~ 1 to keep condition number reasonable
			L_test[i, i] /= 10**np.log(n)
			L_test[i, i] += 1
			# upper triangle = 0
			for j in xrange(n):
				if j > i:
					L_test[i, j] *= 0

		x_rand = np.random.rand(n)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				# make L, x
				L, L_py, L_ptr = self.make_mat_triplet(lib, (n, n),
													   rowmajor=rowmajor)
				x, x_py, x_ptr = self.make_vec_triplet(lib, n)

				# populate L, x
				L_py += L_test
				x_py += x_rand
				lib.vector_memcpy_va(x, x_ptr, 1)
				lib.matrix_memcpy_ma(L, L_ptr, order)

				# y = inv(L) * x
				pyresult = np.linalg.solve(L_test, x_rand)
				lib.blas_trsv(hdl, lib.enums.CblasLower,
							  lib.enums.CblasNoTrans, lib.enums.CblasNonUnit,
							  L, x)
				lib.vector_memcpy_av(x_ptr, x, 1)

				self.assertTrue(np.allclose(x_py, pyresult, DIGITS))

				lib.matrix_free(L)
				lib.vector_free(x)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)


	def test_blas2_sbmv(self):
		(m, n) = self.shape
		diags = max(1, min(4, min(m, n) - 1))

		s_test = np.random.rand(n * diags)
		x_rand = np.random.rand(n)
		y_rand = np.random.rand(n)

		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			# make symmetric banded "matrix" S stored as vector s,
			# and vectors x, y
			s, s_py, s_ptr = self.make_vec_triplet(lib, n * diags)
			x, x_py, x_ptr = self.make_vec_triplet(lib, n)
			y, y_py, y_ptr = self.make_vec_triplet(lib, n)

			# populate vectors
			s_py += s_test
			x_py += x_rand
			y_py += y_rand
			lib.vector_memcpy_va(s, s_ptr, 1)
			lib.vector_memcpy_va(x, x_ptr, 1)
			lib.vector_memcpy_va(y, y_ptr, 1)

			# y = alpha
			alpha = np.random.rand()
			beta = np.random.rand()
			pyresult = np.zeros(n)
			for d in xrange(diags):
				for j in xrange(n - d):
					if d > 0:
						pyresult[d + j] += s_test[d + diags * j] * x_rand[j]
					pyresult[j] += s_test[d + diags * j] * x_rand[d + j]
			pyresult *= alpha
			pyresult += beta * y_py

			lib.blas_sbmv(hdl, lib.enums.CblasColMajor, lib.enums.CblasLower,
						  diags - 1, alpha, s, x, beta, y)
			lib.vector_memcpy_av(y_ptr, y, 1)


			self.assertTrue(np.allclose(y_py, pyresult, DIGITS))

			lib.vector_free(x)
			lib.vector_free(y)
			lib.vector_free(s)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)


	def test_diagmv(self):
		(m, n) = self.shape

		d_test = np.random.rand(n)
		x_rand = np.random.rand(n)
		y_rand = np.random.rand(n)

		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			# make diagonal "matrix" D stored as vector d,
			# and vectors x, y
			d, d_py, d_ptr = self.make_vec_triplet(lib, n)
			x, x_py, x_ptr = self.make_vec_triplet(lib, n)
			y, y_py, y_ptr = self.make_vec_triplet(lib, n)

			# populate vectors
			d_py += d_test
			x_py += x_rand
			y_py += 2
			lib.vector_memcpy_va(d, d_ptr, 1)
			lib.vector_memcpy_va(x, x_ptr, 1)
			lib.vector_memcpy_va(y, y_ptr, 1)

			# y = alpha * D * x + beta * y
			alpha = np.random.rand()
			beta = np.random.rand()
			pyresult = alpha * d_py * x_py + beta * y_py
			lib.blas_diagmv(hdl, alpha, d, x, beta, y)
			lib.vector_memcpy_av(y_ptr, y, 1)

			self.assertTrue(np.allclose(y_py, pyresult, DIGITS))

			lib.vector_free(x)
			lib.vector_free(y)
			lib.vector_free(d)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_blas3_gemm(self):
		(m, n) = self.shape
		x_rand = np.random.rand(n)

		hdl = c_void_p()
		B_test = np.random.rand(m, n)
		C_test = np.random.rand(n, n)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				# make A, x, y
				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n),
													   rowmajor=rowmajor)
				B, B_py, B_ptr = self.make_mat_triplet(lib, (m, n),
													   rowmajor=rowmajor)
				C, C_py, C_ptr = self.make_mat_triplet(lib, (n, n),
													   rowmajor=rowmajor)

				# populate
				A_py += self.A_test
				B_py += B_test
				C_py += C_test
				lib.matrix_memcpy_ma(A, A_ptr, order)
				lib.matrix_memcpy_ma(B, B_ptr, order)
				lib.matrix_memcpy_ma(C, C_ptr, order)

				# perform C = alpha * B'A + beta * C
				alpha = np.random.rand()
				beta = np.random.rand()
				pyresult = alpha * B_py.T.dot(A_py) + beta * C_py
				lib.blas_gemm(hdl, lib.enums.CblasTrans,
							  lib.enums.CblasNoTrans, alpha, B, A, beta, C)
				lib.matrix_memcpy_am(C_ptr, C, order)
				self.assertTrue(np.allclose(pyresult, C_py, DIGITS))

				lib.matrix_free(A)
				lib.matrix_free(B)
				lib.matrix_free(C)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_blas3_syrk(self):
		(m, n) = self.shape

		hdl = c_void_p()
		B_test = np.random.rand(n, n)


		# make B symmetric
		B_test = B_test.T.dot(B_test)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in [True]:
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor


				# make A, x, y
				A, A_py, A_ptr = self.make_mat_triplet(lib, (n, n),
													   rowmajor=rowmajor)
				B, B_py, B_ptr = self.make_mat_triplet(lib, (n, n),
													   rowmajor=rowmajor)

				# populate
				if m >= n:
					A_py += self.A_test[:n, :]
				else:
					A_py[:m, :] += self.A_test
				B_py += B_test
				lib.matrix_memcpy_ma(A, A_ptr, order)
				lib.matrix_memcpy_ma(B, B_ptr, order)

				# B = alpha * (A'A) + beta * B
				alpha = np.random.rand()
				beta = np.random.rand()
				pyresult = alpha * A_py.dot(A_py.T) + beta * B_py
				lib.blas_syrk(hdl, lib.enums.CblasLower,
							  lib.enums.CblasNoTrans, alpha, A, beta, B)
				lib.matrix_memcpy_am(B_ptr, B, order)
				for i in xrange(n):
					for j in xrange(n):
						if j > i:
							pyresult[i, j] *= 0
							B_py[i, j] *= 0
				self.assertTrue(np.allclose(pyresult, B_py, DIGITS))

				lib.matrix_free(A)
				lib.matrix_free(B)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_blas3_trsm(self):
		(m, n) = self.shape
		hdl = c_void_p()

		# make square, invertible L
		L_test = np.random.rand(n, n)

		for i in xrange(n):
			L_test[i, i] /= 10**np.log(n)
			L_test[i, i] += 1
			for j in xrange(n):
				if j > i:
					L_test[i, j]*= 0

		for (gpu, single_precision) in CONDITIONS:
			if gpu:
				continue

			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if single_precision else 7

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				# make A, x, y
				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n),
													   rowmajor=rowmajor)
				L, L_py, L_ptr = self.make_mat_triplet(lib, (n, n),
													   rowmajor=rowmajor)

				# populate
				A_py += self.A_test
				L_py += L_test
				lib.matrix_memcpy_ma(A, A_ptr, order)
				lib.matrix_memcpy_ma(L, L_ptr, order)

				# A = A * inv(L)
				pyresult = A_py.dot(np.linalg.inv(L_test))
				lib.blas_trsm(hdl, lib.enums.CblasRight, lib.enums.CblasLower,
						      lib.enums.CblasNoTrans, lib.enums.CblasNonUnit,
						      1., L, A)
				lib.matrix_memcpy_am(A_ptr, A, order)

				self.assertTrue(np.allclose(pyresult, A_py, DIGITS))

				lib.matrix_free(A)
				lib.matrix_free(L)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

class DenseLinalgTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)

	def test_cholesky(self):
		(m, n) = self.shape
		hdl = c_void_p()

		mindim = min(m, n)

		# build decently conditioned symmetric matrix
		AA_test = self.A_test.T.dot(self.A_test)[:mindim, :mindim]
		AA_test /= np.linalg.norm(AA_test) * mindim**0.5
		for i in xrange(mindim):
			# diagonal entries ~ 1 to keep condition number reasonable
			AA_test[i, i] /= 10**np.log(mindim)
			AA_test[i, i] += 1
			# upper triangle = 0
			for j in xrange(mindim):
				if j > i:
					AA_test[i, j] *= 0
		AA_test += AA_test.T

		x_rand = np.random.rand(mindim)
		pysol = np.linalg.solve(AA_test, x_rand)
		pychol = np.linalg.cholesky(AA_test)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				# allocate L, x
				L = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(L, mindim, mindim, order)
				L_py = np.zeros((mindim, mindim), order=pyorder).astype(
							lib.pyfloat)
				L_ptr = L_py.ctypes.data_as(lib.ok_float_p)

				x = lib.vector(0, 0, None)
				lib.vector_calloc(x, mindim, order)
				x_py = np.zeros(mindim).astype(lib.pyfloat)
				x_ptr = x_py.ctypes.data_as(lib.ok_float_p)

				# populate L
				L_py *= 0
				L_py += AA_test
				lib.matrix_memcpy_ma(L, L_ptr, order)

				# cholesky factorization
				lib.linalg_cholesky_decomp(hdl, L)
				lib.matrix_memcpy_am(L_ptr, L, order)
				for i in xrange(mindim):
					for j in xrange(mindim):
						if j > i:
							L_py[i, j] *= 0

				imprecision_factor = 5**(int(gpu) + int(single_precision))
				atol = 1e-2 * imprecision_factor
				rtol = 1e-2 * imprecision_factor
				norm_diff = np.linalg.norm(
						L_py.dot(x_rand) - pychol.dot(x_rand))
				norm = np.linalg.norm(pychol)
				self.assertTrue(norm_diff <= atol * mindim + rtol * norm)

				# populate x
				x_py *= 0
				x_py += x_rand
				lib.vector_memcpy_va(x, x_ptr, 1)

				# cholesky solve
				lib.linalg_cholesky_svx(hdl, L, x)
				lib.vector_memcpy_av(x_ptr, x, 1)

				norm_diff = np.linalg.norm(x_py - pysol)
				norm = np.linalg.norm(pysol)
				self.assertTrue(norm_diff <= atol * mindim**0.5 + rtol * norm)

				lib.matrix_free(L)
				lib.vector_free(x)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_diag_gramian(self):
		(m, n) = self.shape
		mindim = min(m, n)

		hdl = c_void_p()

		# Python: calculate diag of (AA') (A fat) or (A'A) (A skinny)
		Acols = self.A_test if m >= n else self.A_test.T
		py_diag = np.zeros(mindim)
		for j in xrange(mindim):
			py_diag[j] = Acols[:, j].dot(Acols[:, j])


		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
				single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLMIN = RTOL * mindim**0.5

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				# allocate A, x
				A = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(A, m, n, order)
				A_py = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A_py += self.A_test
				A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_memcpy_ma(A, A_ptr, order)

				x = lib.vector(0, 0, None)
				lib.vector_calloc(x, mindim, order)
				x_py = np.zeros(mindim).astype(lib.pyfloat)
				x_ptr = x_py.ctypes.data_as(lib.ok_float_p)

				# C: calculate diag of (AA') (A fat) or (A'A) (A skinny)
				lib.linalg_diag_gramian(hdl, A, x)
				lib.vector_memcpy_av(x_ptr, x, 1)

				# compare C vs Python results
				# self.assertTrue(np.linalg.norm(x_py - py_diag) <=
								# ATOLMIN + RTOL * np.linalg.norm(py_diag))

				# free memory
				lib.matrix_free(A)
				lib.vector_free(x)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_broadcast(self):
		(m, n) = self.shape
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
				single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				# allocate A, d, e
				A = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(A, m, n, order)
				A_py = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A_py += self.A_test
				A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_memcpy_ma(A, A_ptr, order)

				d = lib.vector(0, 0, None)
				lib.vector_calloc(d, m)
				d_py = np.zeros(m).astype(lib.pyfloat)
				d_ptr = d_py.ctypes.data_as(lib.ok_float_p)

				e = lib.vector(0, 0, None)
				lib.vector_calloc(e, n)
				e_py = np.zeros(n).astype(lib.pyfloat)
				e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

				x = lib.vector(0, 0, None)
				lib.vector_calloc(x, n)
				x_py = np.zeros(n).astype(lib.pyfloat)
				x_ptr = x_py.ctypes.data_as(lib.ok_float_p)

				y = lib.vector(0, 0, None)
				lib.vector_calloc(y, m)
				y_py = np.zeros(m).astype(lib.pyfloat)
				y_ptr = y_py.ctypes.data_as(lib.ok_float_p)

				d_py += np.random.rand(m)
				e_py += np.random.rand(n)
				x_py += np.random.rand(n)
				lib.vector_memcpy_va(d, d_ptr, 1)
				lib.vector_memcpy_va(e, e_ptr, 1)
				lib.vector_memcpy_va(x, x_ptr, 1)

				# A = A * diag(E)
				lib.linalg_matrix_broadcast_vector(hdl, A, e,
					lib.enums.OkTransformScale, lib.enums.CblasRight)

				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				AEx = self.A_test.dot(e_py * x_py)
				self.assertTrue(np.linalg.norm(Ax - AEx) <=
								ATOLM + RTOL * np.linalg.norm(AEx))

				# A = diag(D) * A
				lib.linalg_matrix_broadcast_vector(hdl, A, d,
					lib.enums.OkTransformScale, lib.enums.CblasLeft)
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				DAEx = d_py * AEx
				self.assertTrue(np.linalg.norm(Ax - DAEx) <=
								ATOLM + RTOL * np.linalg.norm(DAEx))

				# A += 1e'
				lib.linalg_matrix_broadcast_vector(hdl, A, e,
					lib.enums.OkTransformAdd, lib.enums.CblasRight)
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				A_updatex = DAEx + np.ones(m) * e_py.dot(x_py)
				self.assertTrue(np.linalg.norm(Ax - A_updatex) <=
								ATOLM + RTOL * np.linalg.norm(A_updatex))

				# A += d1'
				lib.linalg_matrix_broadcast_vector(hdl, A, d,
					lib.enums.OkTransformAdd, lib.enums.CblasLeft)
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				A_updatex += d_py * sum(x_py)
				self.assertTrue(np.linalg.norm(Ax - A_updatex) <=
								ATOLM + RTOL * np.linalg.norm(A_updatex))

				# free memory
				lib.matrix_free(A)
				lib.vector_free(d)
				lib.vector_free(e)
				lib.vector_free(x)
				lib.vector_free(y)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_reduce(self):
		(m, n) = self.shape
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
				single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				# allocate A, d, e
				A = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(A, m, n, order)
				A_py = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A_py += self.A_test
				A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_memcpy_ma(A, A_ptr, order)

				d = lib.vector(0, 0, None)
				lib.vector_calloc(d, m)
				d_py = np.zeros(m).astype(lib.pyfloat)
				d_ptr = d_py.ctypes.data_as(lib.ok_float_p)

				e = lib.vector(0, 0, None)
				lib.vector_calloc(e, n)
				e_py = np.zeros(n).astype(lib.pyfloat)
				e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

				x = lib.vector(0, 0, None)
				lib.vector_calloc(x, n)
				x_py = np.zeros(n).astype(lib.pyfloat)
				x_ptr = x_py.ctypes.data_as(lib.ok_float_p)

				y = lib.vector(0, 0, None)
				lib.vector_calloc(y, m)
				y_py = np.zeros(m).astype(lib.pyfloat)
				y_ptr = y_py.ctypes.data_as(lib.ok_float_p)

				x_py += np.random.rand(n)
				lib.vector_memcpy_va(x, x_ptr, 1)

				# min - reduce columns
				colmin = np.min(A_py, 0)
				lib.linalg_matrix_reduce_min(hdl, e, A, lib.enums.CblasLeft)
				lib.vector_memcpy_av(e_ptr, e, 1)
				self.assertTrue(np.linalg.norm(e_py - colmin) <=
								ATOLN + RTOL * np.linalg.norm(colmin))

				# min - reduce rows
				rowmin = np.min(A_py, 1)
				lib.linalg_matrix_reduce_min(hdl, d, A, lib.enums.CblasRight)
				lib.vector_memcpy_av(d_ptr, d, 1)
				self.assertTrue(np.linalg.norm(d_py - rowmin) <=
								ATOLM + RTOL * np.linalg.norm(rowmin))

				# max - reduce columns
				colmax = np.max(A_py, 0)
				lib.linalg_matrix_reduce_max(hdl, e, A, lib.enums.CblasLeft)
				lib.vector_memcpy_av(e_ptr, e, 1)
				self.assertTrue(np.linalg.norm(e_py - colmax) <=
								ATOLN + RTOL * np.linalg.norm(colmax))

				# max - reduce rows
				rowmax = np.max(A_py, 1)
				lib.linalg_matrix_reduce_max(hdl, d, A, lib.enums.CblasRight)
				lib.vector_memcpy_av(d_ptr, d, 1)
				self.assertTrue(np.linalg.norm(d_py - rowmax) <=
								ATOLM + RTOL * np.linalg.norm(rowmax))

				# indmin - reduce columns
				inds = np.zeros(n).astype(c_size_t)
				inds_ptr = inds.ctypes.data_as(lib.c_size_t_p)
				lib.linalg_matrix_reduce_indmin(hdl, inds_ptr, e, A,
												lib.enums.CblasLeft)
				calcmin = np.array([A_py[inds[i], i] for i in xrange(n)])
				colmin = np.min(A_py, 0)
				print calcmin - colmin
				self.assertTrue(np.linalg.norm(calcmin - colmin) <=
								ATOLN + RTOL * np.linalg.norm(colmin))

				# indmin - reduce rows
				inds = np.zeros(m).astype(c_size_t)
				inds_ptr = inds.ctypes.data_as(lib.c_size_t_p)
				lib.linalg_matrix_reduce_indmin(hdl, inds_ptr, d, A,
												lib.enums.CblasRight)
				calcmin = np.array([A_py[i, inds[i]] for i in xrange(m)])
				rowmin = np.min(A_py, 1)
				print calcmin - rowmin
				self.assertTrue(np.linalg.norm(calcmin - rowmin) <=
											   ATOLM + RTOL * np.linalg.norm(rowmin))

				# free memory
				lib.matrix_free(A)
				lib.vector_free(d)
				lib.vector_free(e)
				lib.vector_free(x)
				lib.vector_free(y)

			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)