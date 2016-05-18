import os
import numpy as np
from ctypes import c_float, c_int, c_size_t, c_void_p, Structure, byref
from optkit.libs.linsys import DenseLinsysLibs
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.base import OptkitCTestCase

class DenseLibsTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(
					single_precision=single_precision, gpu=gpu))
		self.assertTrue( any(libs) )

	def test_lib_types(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertTrue( 'ok_float' in dir(lib) )
			self.assertTrue( 'ok_int' in dir(lib) )
			self.assertTrue( 'c_int_p' in dir(lib) )
			self.assertTrue( 'ok_float_p' in dir(lib) )
			self.assertTrue( 'ok_int_p' in dir(lib) )
			self.assertTrue( 'vector' in dir(lib) )
			self.assertTrue( 'vector_p' in dir(lib) )
			self.assertTrue( 'matrix' in dir(lib) )
			self.assertTrue( 'matrix_p' in dir(lib) )
			self.assertTrue( single_precision == (lib.ok_float == c_float) )
			self.assertTrue( lib.ok_int == c_int )

	def test_blas_handle(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			handle = c_void_p()
			# create
			self.assertCall( lib.blas_make_handle(byref(handle)) )
			# destroy
			self.assertCall( lib.blas_destroy_handle(handle) )

	def test_device_reset(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			# reset
			self.assertCall( lib.ok_device_reset() )

			# allocate - deallocate - reset
			handle = c_void_p()
			self.assertCall( lib.blas_make_handle(byref(handle)) )
			self.assertCall( lib.blas_destroy_handle(handle) )
			self.assertCall( lib.ok_device_reset() )

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
				print("denselib version", version)

class DenseBLASTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = DenseLinsysLibs()
		self.A_test = self.A_test_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_blas1_dot(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			hdl = self.register_blas_handle(lib, 'hdl')
			v, v_py, v_ptr = self.register_vector(lib, m, 'v')
			w, w_py, w_ptr = self.register_vector(lib, m, 'w')

			v_py += np.random.rand(m)
			w_py += np.random.rand(m)
			lib.vector_memcpy_va(v, v_ptr, 1)
			lib.vector_memcpy_va(w, w_ptr, 1)

			answer = np.zeros(1).astype(lib.pyfloat)
			answer_p = answer.ctypes.data_as(lib.ok_float_p)
			self.assertCall( lib.blas_dot(hdl, v, w, answer_p) )
			self.assertTrue( np.abs(answer[0] - v_py.dot(w_py)) <=
							 TOL + TOL * answer[0] )

			self.free_vars('v', 'w', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_blas1_nrm2(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			hdl = self.register_blas_handle(lib, 'hdl')
			v, v_py, v_ptr = self.register_vector(lib, m, 'v')

			v_py += np.random.rand(m)
			lib.vector_memcpy_va(v, v_ptr, 1)

			answer = np.zeros(1).astype(lib.pyfloat)
			answer_p = answer.ctypes.data_as(lib.ok_float_p)
			self.assertCall( lib.blas_nrm2(hdl, v, answer_p) )
			self.assertScalarEqual( answer[0], np.linalg.norm(v_py), TOL )

			self.free_vars('v', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_blas1_asum(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			hdl = self.register_blas_handle(lib, 'hdl')
			v, v_py, v_ptr = self.register_vector(lib, m, 'v')

			v_py += np.random.rand(m)
			self.assertCall( lib.vector_memcpy_va(v, v_ptr, 1) )

			answer = np.zeros(1).astype(lib.pyfloat)
			answer_p = answer.ctypes.data_as(lib.ok_float_p)
			self.assertCall( lib.blas_asum(hdl, v, answer_p) )
			self.assertScalarEqual( answer[0], np.linalg.norm(v_py, 1), TOL )

			self.free_vars('v', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_blas1_scal(self):
		(m, n) = self.shape
		v_rand= np.random.rand(m)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			hdl = self.register_blas_handle(lib, 'hdl')
			v, v_py, v_ptr = self.register_vector(lib, m, 'v')

			v_py += v_rand
			self.assertCall( lib.vector_memcpy_va(v, v_ptr, 1) )

			alpha = np.random.rand()
			self.assertCall( lib.blas_scal(hdl, alpha, v) )
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertVecEqual( v_py, alpha * v_rand, TOL * m**0.5, TOL )

			self.free_vars('v', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_blas1_axpy(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			hdl = self.register_blas_handle(lib, 'hdl')
			v, v_py, v_ptr = self.register_vector(lib, m, 'v')
			w, w_py, w_ptr = self.register_vector(lib, m, 'w')

			v_py += np.random.rand(m)
			w_py += np.random.rand(m)
			alpha = np.random.rand()
			pyresult = alpha * v_py + w_py
			self.assertCall( lib.vector_memcpy_va(v, v_ptr, 1) )
			self.assertCall( lib.vector_memcpy_va(w, w_ptr, 1) )
			self.assertCall( lib.blas_axpy(hdl, alpha, v, w) )
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( w_py, pyresult, TOL * m**0.5, TOL )

			self.free_vars('v', 'w', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_blas2_gemv(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# make A, x, y
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')

				# populate A, x, y (in Py and C)
				A_py += self.A_test
				x_py += np.random.rand(n)
				y_py += np.random.rand(m)
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( lib.vector_memcpy_va(y, y_ptr, 1) )
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				# perform y = alpha * A * x + beta *  y
				alpha = -0.5 + np.random.rand()
				beta = -0.5 + np.random.rand()

				pyresult = alpha * A_py.dot(x_py) + beta * y_py
				self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasNoTrans,
								 alpha, A, x, beta, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				self.assertVecEqual( y_py, pyresult, TOL * m**0.5, TOL )

				# perform x = alpha * A' * y + beta * x
				y_py[:] = pyresult[:]
				pyresult = alpha * A_py.T.dot(y_py) + beta * x_py
				self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasTrans,
								 alpha, A, y, beta, x) )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( x_py, pyresult, TOL * n**0.5, TOL )

				self.free_vars('A', 'x', 'y', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_blas2_trsv(self):
		(m, n) = self.shape

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

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# make L, x
				L, L_py, L_ptr = self.register_matrix(lib, n, n, order, 'L')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')

				# populate L, x
				L_py += L_test
				x_py += x_rand
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( lib.matrix_memcpy_ma(L, L_ptr, order) )

				# y = inv(L) * x
				pyresult = np.linalg.solve(L_test, x_rand)
				self.assertCall( lib.blas_trsv(hdl, lib.enums.CblasLower,
							  lib.enums.CblasNoTrans, lib.enums.CblasNonUnit,
							  L, x) )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( x_py, pyresult, TOL * n**0.5, TOL )

				self.free_vars('L', 'x', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_blas2_sbmv(self):
		(m, n) = self.shape
		diags = max(1, min(4, min(m, n) - 1))

		s_test = np.random.rand(n * diags)
		x_rand = np.random.rand(n)
		y_rand = np.random.rand(n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			hdl = self.register_blas_handle(lib, 'hdl')

			# make symmetric banded "matrix" S stored as vector s,
			# and vectors x, y
			s, s_py, s_ptr = self.register_vector(lib, n * diags, 's')
			x, x_py, x_ptr = self.register_vector(lib, n, 'x')
			y, y_py, y_ptr = self.register_vector(lib, n, 'y')

			# populate vectors
			s_py += s_test
			x_py += x_rand
			y_py += y_rand
			self.assertCall( lib.vector_memcpy_va(s, s_ptr, 1) )
			self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
			self.assertCall( lib.vector_memcpy_va(y, y_ptr, 1) )

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

			self.assertCall( lib.blas_sbmv(hdl, lib.enums.CblasColMajor,
							 lib.enums.CblasLower, diags - 1, alpha, s, x,
							 beta, y) )
			self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
			self.assertVecEqual( y_py, pyresult, TOL * m**0.5, TOL )

			self.free_vars('x', 'y', 's', 'hdl')
			self.assertCall( lib.ok_device_reset() )


	def test_diagmv(self):
		(m, n) = self.shape

		d_test = np.random.rand(n)
		x_rand = np.random.rand(n)
		y_rand = np.random.rand(n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			hdl = self.register_blas_handle(lib, 'hdl')

			# make diagonal "matrix" D stored as vector d,
			# and vectors x, y
			d, d_py, d_ptr = self.register_vector(lib, n, 'd')
			x, x_py, x_ptr = self.register_vector(lib, n, 'x')
			y, y_py, y_ptr = self.register_vector(lib, n, 'y')

			# populate vectors
			d_py += d_test
			x_py += x_rand
			y_py += 2
			self.assertCall( lib.vector_memcpy_va(d, d_ptr, 1) )
			self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
			self.assertCall( lib.vector_memcpy_va(y, y_ptr, 1) )

			# y = alpha * D * x + beta * y
			alpha = np.random.rand()
			beta = np.random.rand()
			pyresult = alpha * d_py * x_py + beta * y_py
			self.assertCall( lib.blas_diagmv(hdl, alpha, d, x, beta, y) )
			self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
			self.assertVecEqual( y_py, pyresult, TOL * m**0.5, TOL )

			self.free_vars('x', 'y', 'd', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_blas3_gemm(self):
		(m, n) = self.shape
		x_rand = np.random.rand(n)

		B_test = np.random.rand(m, n)
		C_test = np.random.rand(n, n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLMN = RTOL * (m * n)**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# allocate A, B
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				B, B_py, B_ptr = self.register_matrix(lib, m, n, order, 'B')
				C, C_py, C_ptr = self.register_matrix(lib, n, n, order, 'C')

				# populate
				A_py += self.A_test
				B_py += B_test
				C_py += C_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
				self.assertCall( lib.matrix_memcpy_ma(B, B_ptr, order) )
				self.assertCall( lib.matrix_memcpy_ma(C, C_ptr, order) )

				# perform C = alpha * B'A + beta * C
				alpha = np.random.rand()
				beta = np.random.rand()
				pyresult = alpha * B_py.T.dot(A_py) + beta * C_py
				self.assertCall( lib.blas_gemm(hdl, lib.enums.CblasTrans,
							  	 lib.enums.CblasNoTrans, alpha, B, A, beta,
							  	 C) )
				self.assertCall( lib.matrix_memcpy_am(C_ptr, C, order) )
				self.assertVecEqual( C_py, pyresult, ATOLMN, RTOL )

				self.free_vars('A', 'B', 'C', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_blas3_syrk(self):
		(m, n) = self.shape
		B_test = np.random.rand(n, n)

		# make B symmetric
		B_test = B_test.T.dot(B_test)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			TOL = 10**(-DIGITS)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# allocate A, B
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				B, B_py, B_ptr = self.register_matrix(lib, n, n, order, 'B')

				# populate
				A_py += self.A_test[:, :]
				B_py += B_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
				self.assertCall( lib.matrix_memcpy_ma(B, B_ptr, order) )

				# B = alpha * (A'A) + beta * B
				alpha = np.random.rand()
				beta = np.random.rand()
				pyresult = alpha * A_py.T.dot(A_py) + beta * B_py
				self.assertCall( lib.blas_syrk(hdl, lib.enums.CblasLower,
							  	 lib.enums.CblasTrans, alpha, A, beta, B) )
				self.assertCall( lib.matrix_memcpy_am(B_ptr, B, order) )
				for i in xrange(n):
					for j in xrange(n):
						if j > i:
							pyresult[i, j] *= 0
							B_py[i, j] *= 0
				self.assertVecEqual( B_py, pyresult, TOL * n, TOL )

				self.free_vars('A', 'B', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_blas3_trsm(self):
		(m, n) = self.shape

		# make square, invertible L
		L_test = np.random.rand(n, n)

		for i in xrange(n):
			L_test[i, i] /= 10**np.log(n)
			L_test[i, i] += 1
			for j in xrange(n):
				if j > i:
					L_test[i, j]*= 0

		for (gpu, single_precision) in self.CONDITIONS:
			if gpu:
				continue

			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLMN = RTOL * (m * n)**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# allocate A, L
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				L, L_py, L_ptr = self.register_matrix(lib, n, n, order, 'L')

				# populate
				A_py += self.A_test
				L_py += L_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
				self.assertCall( lib.matrix_memcpy_ma(L, L_ptr, order) )

				# A = A * inv(L)
				pyresult = A_py.dot(np.linalg.inv(L_test))
				self.assertCall( lib.blas_trsm(hdl, lib.enums.CblasRight,
								 lib.enums.CblasLower, lib.enums.CblasNoTrans,
								 lib.enums.CblasNonUnit, 1., L, A) )
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, pyresult, ATOLMN, RTOL )

				self.free_vars('A', 'L', 'hdl')
				self.assertCall( lib.ok_device_reset() )

class DenseLinalgTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = DenseLinsysLibs()
		self.A_test = self.A_test_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_cholesky(self):
		(m, n) = self.shape

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

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# allocate L, x
				L, L_py, L_ptr = self.register_matrix(
					lib, mindim, mindim, order, 'L')
				x, x_py, x_ptr = self.register_vector(lib, mindim, 'x')

				# populate L
				L_py *= 0
				L_py += AA_test
				self.assertCall( lib.matrix_memcpy_ma(L, L_ptr, order) )

				# cholesky factorization
				self.assertCall( lib.linalg_cholesky_decomp(hdl, L) )
				self.assertCall( lib.matrix_memcpy_am(L_ptr, L, order) )
				for i in xrange(mindim):
					for j in xrange(mindim):
						if j > i:
							L_py[i, j] *= 0

				imprecision_factor = 5**(int(gpu) + int(single_precision))
				atol = 1e-2 * imprecision_factor * mindim
				rtol = 1e-2 * imprecision_factor
				self.assertVecEqual( L_py.dot(x_rand), pychol.dot(x_rand),
									 atol, rtol )

				# populate x
				x_py *= 0
				x_py += x_rand
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

				# cholesky solve
				self.assertCall( lib.linalg_cholesky_svx(hdl, L, x) )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( x_py, pysol, atol * mindim**0.5, rtol )

				self.free_vars('L', 'x', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_row_squares(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				# allocate A, r, c
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				c, c_py, c_ptr = self.register_vector(lib, n, 'c')
				r, r_py, r_ptr = self.register_vector(lib, m, 'r')

				A_py += self.A_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				py_rows = [A_py[i, :].dot(A_py[i, :]) for i in xrange(m)]
				py_cols = [A_py[:, j].dot(A_py[:, j]) for j in xrange(n)]

				# C: calculate row squares
				self.assertCall( lib.linalg_matrix_row_squares(
						lib.enums.CblasNoTrans, A, r) )
				self.assertCall( lib.vector_memcpy_av(r_ptr, r, 1) )

				# compare C vs Python results
				self.assertVecEqual( r_py, py_rows, ATOLM, RTOL )

				# C: calculate column squares
				self.assertCall( lib.linalg_matrix_row_squares(
						lib.enums.CblasTrans, A, c) )
				self.assertCall( lib.vector_memcpy_av(c_ptr, c, 1) )

				# compare C vs Python results
				self.assertVecEqual( c_py, py_cols, ATOLN, RTOL )

				# free memory
				self.free_vars('A', 'r', 'c')
				self.assertCall( lib.ok_device_reset() )

	def test_broadcast(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# allocate A, d, e, x, y
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				d, d_py, d_ptr = self.register_vector(lib, m, 'd')
				e, e_py, e_ptr = self.register_vector(lib, n, 'e')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')

				A_py += self.A_test
				d_py += np.random.rand(m)
				e_py += np.random.rand(n)
				x_py += np.random.rand(n)
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
				self.assertCall( lib.vector_memcpy_va(d, d_ptr, 1) )
				self.assertCall( lib.vector_memcpy_va(e, e_ptr, 1) )
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

				# A = A * diag(E)
				self.assertCall( lib.linalg_matrix_broadcast_vector(A, e,
										lib.enums.OkTransformScale,
										lib.enums.CblasRight) )

				self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1,
										A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				Ax = y_py
				AEx = A_py.dot(e_py * x_py)

				self.assertVecEqual( Ax, AEx, ATOLM, RTOL )

				# A = diag(D) * A
				self.assertCall( lib.linalg_matrix_broadcast_vector(A, d,
										lib.enums.OkTransformScale,
										lib.enums.CblasLeft) )
				self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1,
										A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				Ax = y_py
				DAEx = d_py * AEx
				self.assertVecEqual( Ax, DAEx, ATOLM, RTOL )

				# A += 1e'
				self.assertCall( lib.linalg_matrix_broadcast_vector(A, e,
										lib.enums.OkTransformAdd,
										lib.enums.CblasRight) )
				self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1,
										A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				Ax = y_py
				A_updatex = DAEx + np.ones(m) * e_py.dot(x_py)
				self.assertVecEqual( Ax, A_updatex, ATOLM, RTOL )

				# A += d1'
				self.assertCall( lib.linalg_matrix_broadcast_vector(A, d,
										lib.enums.OkTransformAdd,
										lib.enums.CblasLeft) )
				self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1,
										A, x, 0, y) )
				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				Ax = y_py
				A_updatex += d_py * sum(x_py)
				self.assertVecEqual( Ax, A_updatex, ATOLM, RTOL )

				# free memory
				self.free_vars('A', 'd', 'e', 'x', 'y', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_reduce(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				# allocate A, d, e, x, y
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				d, d_py, d_ptr = self.register_vector(lib, m, 'd')
				e, e_py, e_ptr = self.register_vector(lib, n, 'e')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')

				A_py += self.A_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				x_py += np.random.rand(n)
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

				# min - reduce columns
				colmin = np.min(A_py, 0)
				self.assertCall( lib.linalg_matrix_reduce_min(e, A,
										lib.enums.CblasLeft) )
				self.assertCall( lib.vector_memcpy_av(e_ptr, e, 1) )
				self.assertVecEqual( e_py, colmin, ATOLN, RTOL )

				# min - reduce rows
				rowmin = np.min(A_py, 1)
				self.assertCall( lib.linalg_matrix_reduce_min(d, A,
										lib.enums.CblasRight) )
				self.assertCall( lib.vector_memcpy_av(d_ptr, d, 1) )
				self.assertVecEqual( d_py, rowmin, ATOLM, RTOL )

				# max - reduce columns
				colmax = np.max(A_py, 0)
				self.assertCall( lib.linalg_matrix_reduce_max(e, A,
										lib.enums.CblasLeft) )
				self.assertCall( lib.vector_memcpy_av(e_ptr, e, 1) )
				self.assertVecEqual( e_py, colmax, ATOLN, RTOL )

				# max - reduce rows
				rowmax = np.max(A_py, 1)
				self.assertCall( lib.linalg_matrix_reduce_max(d, A,
										lib.enums.CblasRight) )
				self.assertCall( lib.vector_memcpy_av(d_ptr, d, 1) )
				self.assertVecEqual( d_py, rowmax, ATOLM, RTOL )

				# indmin - reduce columns
				idx, inds, inds_ptr = self.register_indvector(lib, n, 'idx')
				self.assertCall( lib.linalg_matrix_reduce_indmin(idx, e, A,
										lib.enums.CblasLeft) )
				self.assertCall( lib.indvector_memcpy_av(inds_ptr, idx, 1) )
				self.free_var('idx')
				calcmin = np.array([A_py[inds[i], i] for i in xrange(n)])
				colmin = np.min(A_py, 0)
				self.assertVecEqual( calcmin, colmin, ATOLN, RTOL )

				# indmin - reduce rows
				idx, inds, inds_ptr = self.register_indvector(lib, m, 'idx')
				self.assertCall( lib.linalg_matrix_reduce_indmin(idx, d, A,
										lib.enums.CblasRight) )
				self.assertCall( lib.indvector_memcpy_av(inds_ptr, idx, 1) )
				self.free_var('idx')
				calcmin = np.array([A_py[i, inds[i]] for i in xrange(m)])
				rowmin = np.min(A_py, 1)
				self.assertVecEqual( calcmin, rowmin, ATOLM, RTOL )

				# free memory
				self.free_vars('A', 'd', 'e', 'x', 'y', 'hdl')
				self.assertCall(lib.ok_device_reset() )