import unittest
import os
import numpy as np
from numpy import ndarray
from ctypes import c_void_p, c_size_t, byref
from optkit.libs import DenseLinsysLibs
from optkit.libs.clustering import ClusteringLibs
from optkit.libs.error import optkit_print_error as PRINTERR
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
from optkit.tests.C.base import OptkitCTestCase

class ClusterLibsTestCase(OptkitCTestCase):
	"""
		Equilibrate input A_in as

			D * A_equil * E

		with D, E, diagonal.

		Test that

			D * A_equil * E == A_in,

		or that

			D^-1 * A_in * E^-1 == A_equil.
	"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.cluster_libs = ClusteringLibs()

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

		self.x_test = np.random.rand(self.shape[1])
		self.k = self.shape[0] / 3
		self.u_test = (self.k * np.random.rand(self.shape[0])).astype(c_size_t)

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()

	def test_libs_exist(self):
		dlibs = []
		cluslibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(
					gpu=gpu, single_precision=single_precision))
			cluslibs.append(self.cluster_libs.get(
					dlibs[-1], gpu=gpu, single_precision=single_precision))

		self.assertTrue(any(dlibs))
		self.assertTrue(any(cluslibs))

	@staticmethod
	def upsamplingvec_mul(tU, tA, tB, alpha, u, A, beta, B):
		if not bool(isinstance(u, ndarray) and isinstance(A, ndarray)
					and isinstance(B, ndarray)):
			raise TypeError('u, A, and B must be of type {}'.format(ndarray))
		if not (len(A.shape) == 2 and len(B.shape) == 2):
			raise ValueError('A and B must be 2-d {}s'.format(ndarray))

		umax = int(u.max()) + 1
		u_dim1 = umax if tU == 'T' else len(u)
		u_dim2 = len(u) if tU == 'T' else umax
		A = A.T if tA == 'T' else A
		B = B.T if tB == 'T' else B

		if not bool(A.shape[1] == B.shape[1] and B.shape[0] == u_dim1 and
					u_dim2 <= A.shape[0]):
			raise ValueError('incompatible dimensions')

		B *= beta

		if tU == 'T':
			for (row_a, row_b) in enumerate(u):
				B[row_b, :] += alpha * A[row_a, :]
		else:
			for (row_b, row_a) in enumerate(u):
				B[row_b, :] += alpha * A[row_a, :]

	@staticmethod
	def cluster(A, C, a2c, maxdist):
		if not bool(isinstance(a2c, ndarray) and isinstance(A, ndarray)
					and isinstance(C, ndarray)):
			raise TypeError('a2c, A, and C must be of type {}'.format(ndarray))
		if not (len(A.shape) == 2 and len(C.shape) == 2):
			raise ValueError('A and C must be 2-d {}s'.format(ndarray))
		if not bool(A.shape[1] == C.shape[1] and A.shape[0] == len(a2c) and
					a2c.max() <= C.shape[0]):
			raise ValueError('incompatible dimensions')

		m, n = A.shape
		k = C.shape[0]

		c_squared = np.zeros(k)
		for row, c in enumerate(C):
			c_squared[row] = c.dot(c)

		D = - 2 * C.dot(A.T)
		for i in xrange(m):
			D[:, i] += c_squared

		indmin = D.argmin(0)
		dmin = D.min(0)

		reassigned = 0
		if maxdist == np.inf:
			reassigned = sum(a2c != indmin)
			a2c[:] = indmin[:]
		else:
			for i in xrange(m):
				if a2c[i] == indmin[i]:
					continue
				elif np.abs(A[i, :] - C[indmin[i], :]).max() <= maxdist:
					a2c[i] = indmin[i]
					reassigned += 1

		return D, dmin, reassigned

	def test_upsamplingvec_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			u = lib.upsamplingvec()

			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)

			self.assertEqual( err, 0 )
			self.assertEqual( u.size1, m )
			self.assertEqual( u.size2, k )

			err = PRINTERR( lib.upsamplingvec_free(u) )
			self.unregister_var('u')

			self.assertEqual( err, 0 )

	def test_upsamplingvec_check_bounds(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			u_py = np.zeros(m).astype(c_size_t)
			u_ptr = u_py.ctypes.data_as(dlib.c_size_t_p)

			u = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			self.assertEqual( lib.upsamplingvec_check_bounds(u), 0 )
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)
			self.assertEqual( lib.upsamplingvec_check_bounds(u), 0 )
			u_py += 2 * k
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)
			self.assertNotEqual( lib.upsamplingvec_check_bounds(u), 0 )

			err = PRINTERR( lib.upsamplingvec_free(u) )
			self.assertEqual( err, 0 )
			self.unregister_var('u')

	def test_upsamplingvec_subvector(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			offset = m / 4
			k_offset = k / 4
			msub = m / 2
			ksub = k / 2

			u_py = np.zeros(m).astype(c_size_t)
			u_ptr = u_py.ctypes.data_as(dlib.c_size_t_p)
			usub_py = np.zeros(m / 2).astype(c_size_t)
			usub_ptr = usub_py.ctypes.data_as(dlib.c_size_t_p)

			u = lib.upsamplingvec()
			usub = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			u_py += (k * np.random.rand(m)).astype(c_size_t)
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)
			lib.upsamplingvec_subvector(usub, u, offset, 0, msub, k)
			dlib.indvector_memcpy_av(usub_ptr, usub.vec, 1)
			self.assertTrue( usub.size1 == msub )
			self.assertTrue( usub.size2 == k )
			self.assertTrue( sum(usub_py - u_py[offset : offset + msub]) == 0 )

			lib.upsamplingvec_subvector(usub, u, offset, k_offset, msub, ksub)
			self.assertTrue(usub.size1 == msub)
			self.assertTrue(usub.size2 == ksub - k_offset)

			err = PRINTERR( lib.upsamplingvec_free(u) )
			self.assertEqual( err, 0 )
			self.unregister_var('u')

	def test_upsamplingvec_mul_matrix(self):
		m, n = self.shape
		k = self.k
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			err = dlib.blas_make_handle(byref(hdl))
			self.register_var('hdl', hdl, dlib.blas_destroy_handle)
			self.assertEqual( err, 0 )

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5
			ATOLK = RTOL * k**0.5

			A = dlib.matrix(0, 0, 0, None, 0)
			B = dlib.matrix(0, 0, 0, None, 0)
			C = dlib.matrix(0, 0, 0, None, 0)
			D = dlib.matrix(0, 0, 0, None, 0)
			E = dlib.matrix(0, 0, 0, None, 0)
			dlib.matrix_calloc(A, m, n, dlib.enums.CblasRowMajor)
			dlib.matrix_calloc(B, n, m, dlib.enums.CblasRowMajor)
			dlib.matrix_calloc(C, k, n, dlib.enums.CblasRowMajor)
			dlib.matrix_calloc(D, n, k, dlib.enums.CblasRowMajor)
			self.register_var('A', A, dlib.matrix_free)
			self.register_var('B', B, dlib.matrix_free)
			self.register_var('C', C, dlib.matrix_free)
			self.register_var('D', D, dlib.matrix_free)

			A_py = np.random.rand(m, n).astype(dlib.pyfloat)
			B_py = np.random.rand(n, m).astype(dlib.pyfloat)
			C_py = np.random.rand(k, n).astype(dlib.pyfloat)
			D_py = np.random.rand(n, k).astype(dlib.pyfloat)
			A_ptr = A_py.ctypes.data_as(dlib.ok_float_p)
			B_ptr = B_py.ctypes.data_as(dlib.ok_float_p)
			C_ptr = C_py.ctypes.data_as(dlib.ok_float_p)
			D_ptr = D_py.ctypes.data_as(dlib.ok_float_p)
			dlib.matrix_memcpy_ma(A, A_ptr, dlib.enums.CblasRowMajor)
			dlib.matrix_memcpy_ma(B, B_ptr, dlib.enums.CblasRowMajor)
			dlib.matrix_memcpy_ma(C, C_ptr, dlib.enums.CblasRowMajor)
			dlib.matrix_memcpy_ma(D, D_ptr, dlib.enums.CblasRowMajor)

			mvec = dlib.vector()
			nvec = dlib.vector()
			kvec = dlib.vector()
			dlib.vector_calloc(mvec, m)
			dlib.vector_calloc(nvec, n)
			dlib.vector_calloc(kvec, k)
			self.register_var('mvec', mvec, dlib.vector_free)
			self.register_var('nvec', nvec, dlib.vector_free)
			self.register_var('kvec', kvec, dlib.vector_free)

			mvec_py = np.random.rand(m).astype(dlib.pyfloat)
			nvec_py = np.random.rand(n).astype(dlib.pyfloat)
			kvec_py = np.random.rand(k).astype(dlib.pyfloat)
			mvec_ptr = mvec_py.ctypes.data_as(dlib.ok_float_p)
			nvec_ptr = nvec_py.ctypes.data_as(dlib.ok_float_p)
			kvec_ptr = kvec_py.ctypes.data_as(dlib.ok_float_p)
			dlib.vector_memcpy_va(mvec, mvec_ptr, 1)
			dlib.vector_memcpy_va(nvec, nvec_ptr, 1)
			dlib.vector_memcpy_va(kvec, kvec_ptr, 1)

			u = lib.upsamplingvec()
			usub = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			u_py = self.u_test
			u_py[-1] = self.k - 1
			u_ptr = u_py.ctypes.data_as(dlib.c_size_t_p)
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)

			T = dlib.enums.CblasTrans
			N = dlib.enums.CblasNoTrans
			alpha = np.random.rand()
			# beta = np.random.rand()
			beta = 0

			# functioning cases
			self.upsamplingvec_mul('N', 'N', 'N', alpha, u_py, C_py, beta,
				A_py)
			result_py = A_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(N, N, N, alpha, u, C, beta, A)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, A, nvec, 0, mvec)
			dlib.vector_memcpy_av(mvec_ptr, mvec, 1)
			self.assertTrue( np.linalg.norm(mvec_py - result_py) <=
							 ATOLM + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'N', 'N', alpha, u_py, A_py, beta,
				C_py)
			result_py = C_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(T, N, N, alpha, u, A, beta, C)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, C, nvec, 0, kvec)
			dlib.vector_memcpy_av(kvec_ptr, kvec, 1)
			self.assertTrue( np.linalg.norm(kvec_py - result_py) <=
							 ATOLK + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('N', 'N', 'T', alpha, u_py, C_py, beta,
				B_py)
			result_py = B_py.dot(mvec_py)
			err = lib.upsamplingvec_mul_matrix(N, N, T, alpha, u, C, beta, B)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, B, mvec, 0, nvec)
			dlib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'N', 'T', alpha, u_py, A_py, beta, D_py)
			result_py = D_py.dot(kvec_py)
			err = lib.upsamplingvec_mul_matrix(T, N, T, alpha, u, A, beta, D)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, D, kvec, 0, nvec)
			dlib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('N', 'T', 'N', alpha, u_py, D_py, beta,
				A_py)
			result_py = A_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(N, T, N, alpha, u, D, beta, A)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, A, nvec, 0, mvec)
			dlib.vector_memcpy_av(mvec_ptr, mvec, 1)
			self.assertTrue( np.linalg.norm(mvec_py - result_py) <=
							 ATOLM + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'T', 'N', alpha, u_py, B_py, beta,
				C_py)
			result_py = C_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(T, T, N, alpha, u, B, beta, C)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, C, nvec, 0, kvec)
			dlib.vector_memcpy_av(kvec_ptr, kvec, 1)
			self.assertTrue( np.linalg.norm(kvec_py - result_py) <=
							 ATOLK + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('N', 'T', 'T', alpha, u_py, D_py, beta, B_py)
			result_py = B_py.dot(mvec_py)
			err = lib.upsamplingvec_mul_matrix(N, T, T, alpha, u, D, beta, B)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, B, mvec, 0, nvec)
			dlib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'T', 'T', alpha, u_py, B_py, beta, D_py)
			result_py = D_py.dot(kvec_py)
			err = lib.upsamplingvec_mul_matrix(T, T, T, alpha, u, B, beta, D)
			self.assertEqual( PRINTERR(err), 0 )
			dlib.blas_gemv(hdl, N, 1, D, kvec, 0, nvec)
			dlib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			# reject: dimension mismatch
			err = lib.upsamplingvec_mul_matrix(N, N, N, alpha, u, C, beta, B)
			self.assertEqual( err, 101L )
			err = lib.upsamplingvec_mul_matrix(T, N, N, alpha, u, A, beta, D)
			self.assertEqual( err, 101L )

			# reject: unallocated
			err = lib.upsamplingvec_mul_matrix(T, N, N, alpha, u, E, beta, C)
			self.assertEqual( err, 10L )

			self.free_var('A')
			self.free_var('B')
			self.free_var('C')
			self.free_var('D')
			self.free_var('mvec')
			self.free_var('nvec')
			self.free_var('kvec')
			self.free_var('u')
			self.free_var('hdl')

			self.assertEqual( dlib.ok_device_reset(), 0 )

	def test_upsamplingvec_count(self):
		m, n = self.shape
		k = self.k
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			u = lib.upsamplingvec()
			usub = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			counts = dlib.vector()
			dlib.vector_calloc(counts, k)
			self.register_var('counts', counts, dlib.vector_free)
			counts_py = np.zeros(k).astype(dlib.pyfloat)
			counts_ptr = counts_py.ctypes.data_as(dlib.ok_float_p)

			u_py = self.u_test
			u_py[-1] = self.k - 1
			u_ptr = u_py.ctypes.data_as(dlib.c_size_t_p)
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)

			lib.upsamplingvec_count(u, counts)
			dlib.vector_memcpy_av(counts_ptr, counts, 1)

			counts_host = np.zeros(k)
			for idx in u_py:
				counts_host[idx] += 1

			self.assertTrue( sum(counts_py - counts_host) < 1e-7 * k**0.5 )

			self.free_var('u')
			self.free_var('counts')

	def test_upsamplingvec_shift(self):
		m, n = self.shape
		k = self.k
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			u = lib.upsamplingvec()
			usub = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			u_py = self.u_test
			u_py[-1] = self.k - 1
			u_ptr = u_py.ctypes.data_as(dlib.c_size_t_p)
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)

			u_orig = np.zeros(m).astype(c_size_t)
			u_orig += u_py

			shift = int(k * np.random.rand())

			u_orig += shift
			lib.upsamplingvec_shift(u, shift, dlib.enums.OkTransformIncrement)
			dlib.indvector_memcpy_av(u_ptr, u.vec, 1)
			self.assertTrue( sum(u_orig - u_py) < 1e-7 * m**0.5 )

			u_orig -= shift
			lib.upsamplingvec_shift(u, shift, dlib.enums.OkTransformDecrement)
			dlib.indvector_memcpy_av(u_ptr, u.vec, 1)
			self.assertTrue( sum(u_orig - u_py) < 1e-7 * m**0.5 )

			self.free_var('u')

	def test_cluster_aid_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			h = lib.cluster_aid()

			err = PRINTERR( lib.cluster_aid_alloc(h, m, k,
				dlib.enums.CblasRowMajor) )
			self.register_var('h', h, lib.cluster_aid_free)

			self.assertEqual( err, 0 )
			self.assertEqual( h.a2c_tentative_full.size1, m )
			self.assertEqual( h.a2c_tentative_full.size2, k )
			self.assertEqual( h.a2c_tentative.size1, m )
			self.assertEqual( h.a2c_tentative.size2, k )
			self.assertEqual( h.d_min_full.size, m )
			self.assertEqual( h.d_min.size, m )
			self.assertEqual( h.c_squared_full.size, k )
			self.assertEqual( h.c_squared.size, k )
			self.assertEqual( h.D_full.size1, k )
			self.assertEqual( h.D_full.size2, m )
			self.assertEqual( h.D.size1, k )
			self.assertEqual( h.D.size2, m )
			self.assertEqual( h.reassigned, 0 )

			err = PRINTERR( lib.cluster_aid_free(h) )
			self.unregister_var('h')

			self.assertEqual( h.a2c_tentative_full.size1, 0 )
			self.assertEqual( h.a2c_tentative_full.size2, 0 )
			self.assertEqual( h.d_min_full.size, 0 )
			self.assertEqual( h.c_squared_full.size, 0 )
			self.assertEqual( h.D_full.size1, 0 )
			self.assertEqual( h.D_full.size2, 0 )
			self.assertEqual( err, 0 )

	def test_cluster(self):
		""" cluster

			given matrix A, centroid matrix C, upsampling vector a2c,

			update cluster assignments in vector a2c, based on pairwise
			euclidean distances between rows of A and C

			compare # of reassignments

			let D be the matrix of pairwise distances and dmin be the
			column-wise minima of D:
				-compare D * xrand in Python vs. C
				-compare dmin in Python vs. C

		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			for MAXDIST in [1e3, 0.2]:

				hdl = c_void_p()
				err = dlib.blas_make_handle(byref(hdl))
				self.register_var('hdl', hdl, dlib.blas_destroy_handle)
				self.assertEqual( err, 0 )

				DIGITS = 7 - 2 * single_precision - 1 * gpu
				RTOL = 10**(-DIGITS)
				ATOLM = RTOL * m**0.5
				ATOLN = RTOL * n**0.5
				ATOLK = RTOL * k**0.5

				A = dlib.matrix(0, 0, 0, None, 0)
				C = dlib.matrix(0, 0, 0, None, 0)
				dlib.matrix_calloc(A, m, n, dlib.enums.CblasRowMajor)
				dlib.matrix_calloc(C, k, n, dlib.enums.CblasRowMajor)
				self.register_var('A', A, dlib.matrix_free)
				self.register_var('C', C, dlib.matrix_free)

				A_py = np.random.rand(m, n).astype(dlib.pyfloat)
				C_py = np.random.rand(k, n).astype(dlib.pyfloat)
				A_ptr = A_py.ctypes.data_as(dlib.ok_float_p)
				C_ptr = C_py.ctypes.data_as(dlib.ok_float_p)
				dlib.matrix_memcpy_ma(A, A_ptr, dlib.enums.CblasRowMajor)
				dlib.matrix_memcpy_ma(C, C_ptr, dlib.enums.CblasRowMajor)

				a2c = lib.upsamplingvec()
				err = PRINTERR( lib.upsamplingvec_alloc(a2c, m, k) )
				self.register_var('a2c', a2c, lib.upsamplingvec_free)
				self.assertEqual( err, 0 )

				a2c_py = np.zeros(m).astype(c_size_t)
				a2c_ptr = a2c_py.ctypes.data_as(dlib.c_size_t_p)

				a2c_py += self.u_test
				dlib.indvector_memcpy_va(a2c.vec, a2c_ptr, 1)

				h = lib.cluster_aid_p()

				D, dmin, reassigned = self.cluster(A_py, C_py, a2c_py, MAXDIST)

				lib.cluster(A, C, a2c, byref(h), MAXDIST)
				self.register_var('h', h, lib.cluster_aid_free)

				# compare number of reassignments, C vs Py
				ATOL_REASSIGN = int(1 + 0.1 * k)
				RTOL_REASSIGN = int(1 + 0.1 * reassigned)
				self.assertTrue( abs(h.contents.reassigned - reassigned) <=
								 ATOL_REASSIGN + RTOL_REASSIGN )

				# verify number reassigned
				dlib.indvector_memcpy_av(a2c_ptr, a2c.vec, 1)
				self.assertEqual( h.contents.reassigned,
					sum(a2c_py != self.u_test) )

				# verify all distances
				kvec = dlib.vector()
				dlib.vector_calloc(kvec, k)
				self.register_var('kvec', kvec, dlib.vector_free)

				mvec = dlib.vector()
				dlib.vector_calloc(mvec, m)
				self.register_var('mvec', mvec, dlib.vector_free)

				kvec_py = np.zeros(k).astype(dlib.pyfloat)
				kvec_ptr = kvec_py.ctypes.data_as(dlib.ok_float_p)

				mvec_py = np.random.rand(m).astype(dlib.pyfloat)
				mvec_ptr = mvec_py.ctypes.data_as(dlib.ok_float_p)

				dlib.vector_memcpy_va(mvec, mvec_ptr, 1)
				dlib.blas_gemv(hdl, dlib.enums.CblasNoTrans, 1,
							   h.contents.D_full, mvec, 0, kvec)
				dlib.vector_memcpy_av(kvec_ptr, kvec, 1)

				Dmvec = D.dot(mvec_py)
				self.assertTrue( np.linalg.norm(Dmvec - kvec_py) <=
								 ATOLK + RTOL * np.linalg.norm(Dmvec) )

				# verify min distances
				dmin_py = np.zeros(m).astype(dlib.pyfloat)
				dmin_ptr = dmin_py.ctypes.data_as(dlib.ok_float_p)
				dlib.vector_memcpy_av(dmin_ptr, h.contents.d_min_full, 1)
				self.assertTrue( np.linalg.norm(dmin_py - dmin) <=
								 ATOLM + RTOL * np.linalg.norm(dmin) )

				err = PRINTERR( lib.cluster_aid_free(h) )
				self.unregister_var('h')

				self.free_var('A')
				self.free_var('C')
				self.free_var('a2c')
				self.free_var('mvec')
				self.free_var('kvec')
				self.free_var('hdl')


				self.assertEqual( dlib.ok_device_reset(), 0 )

	def test_calculate_centroids(self):
		""" calculate centroids

			given matrix A, upsampling vector a2c,
			calculate centroid matrix C

			compare C * xrand in Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			hdl = c_void_p()
			err = dlib.blas_make_handle(byref(hdl))
			self.register_var('hdl', hdl, dlib.blas_destroy_handle)
			self.assertEqual( err, 0 )

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5
			ATOLK = RTOL * k**0.5

			A = dlib.matrix(0, 0, 0, None, 0)
			C = dlib.matrix(0, 0, 0, None, 0)
			dlib.matrix_calloc(A, m, n, dlib.enums.CblasRowMajor)
			dlib.matrix_calloc(C, k, n, dlib.enums.CblasRowMajor)
			self.register_var('A', A, dlib.matrix_free)
			self.register_var('C', C, dlib.matrix_free)

			A_py = np.random.rand(m, n).astype(dlib.pyfloat)
			C_py = np.random.rand(k, n).astype(dlib.pyfloat)
			A_ptr = A_py.ctypes.data_as(dlib.ok_float_p)
			C_ptr = C_py.ctypes.data_as(dlib.ok_float_p)
			dlib.matrix_memcpy_ma(A, A_ptr, dlib.enums.CblasRowMajor)
			dlib.matrix_memcpy_ma(C, C_ptr, dlib.enums.CblasRowMajor)

			a2c = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(a2c, m, k) )
			self.register_var('a2c', a2c, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			a2c_py = np.zeros(m).astype(c_size_t)
			a2c_ptr = a2c_py.ctypes.data_as(dlib.c_size_t_p)

			a2c_py += self.u_test
			dlib.indvector_memcpy_va(a2c.vec, a2c_ptr, 1)

			counts = dlib.vector()
			dlib.vector_calloc(counts, k)
			self.register_var('counts', counts, dlib.vector_free)

			kvec = dlib.vector()
			dlib.vector_calloc(kvec, k)
			self.register_var('kvec', kvec, dlib.vector_free)

			nvec = dlib.vector()
			dlib.vector_calloc(nvec, n)
			self.register_var('nvec', nvec, dlib.vector_free)

			kvec_py = np.zeros(k).astype(dlib.pyfloat)
			kvec_ptr = kvec_py.ctypes.data_as(dlib.ok_float_p)

			nvec_py = np.random.rand(n).astype(dlib.pyfloat)
			nvec_ptr = nvec_py.ctypes.data_as(dlib.ok_float_p)

			# C: build centroids
			lib.calculate_centroids(A, C, a2c, counts)

			# Py: build centroids
			self.upsamplingvec_mul('T', 'N', 'N', 1, a2c_py, A_py, 0, C_py)
			counts_local = np.zeros(k)
			for c in a2c_py:
				counts_local[c] += 1
			for idx, c in enumerate(counts_local):
				counts_local[idx] = (1. / c) if c > 0 else 0.
				C_py[idx, :] *= counts_local[idx]

			# C: kvec = C * nvec
			dlib.vector_memcpy_va(nvec, nvec_ptr, 1)
			dlib.blas_gemv(hdl, dlib.enums.CblasNoTrans, 1, C, nvec, 0,
						   kvec)
			dlib.vector_memcpy_av(kvec_ptr, kvec, 1)

			# Py: kvec = C * nvec
			Cnvec = C_py.dot(nvec_py)

			# compare C vs. Py
			self.assertTrue( np.linalg.norm(kvec_py - Cnvec) <=
							 ATOLK + RTOL * np.linalg.norm(Cnvec) )

			self.free_var('A')
			self.free_var('C')
			self.free_var('a2c')
			self.free_var('counts')
			self.free_var('nvec')
			self.free_var('kvec')
			self.free_var('hdl')

			self.assertEqual( dlib.ok_device_reset(), 0 )

	def test_kmeans(self):
		""" k-means

			given matrix A, cluster # k

			cluster A by kmeans algorithm. compare Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			hdl = c_void_p()
			err = dlib.blas_make_handle(byref(hdl))
			self.register_var('hdl', hdl, dlib.blas_destroy_handle)
			self.assertEqual( err, 0 )

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5
			ATOLK = RTOL * k**0.5

			A = dlib.matrix(0, 0, 0, None, 0)
			C = dlib.matrix(0, 0, 0, None, 0)
			dlib.matrix_calloc(A, m, n, dlib.enums.CblasRowMajor)
			dlib.matrix_calloc(C, k, n, dlib.enums.CblasRowMajor)
			self.register_var('A', A, dlib.matrix_free)
			self.register_var('C', C, dlib.matrix_free)

			A_py = np.random.rand(m, n).astype(dlib.pyfloat)
			C_py = np.random.rand(k, n).astype(dlib.pyfloat)
			A_ptr = A_py.ctypes.data_as(dlib.ok_float_p)
			C_ptr = C_py.ctypes.data_as(dlib.ok_float_p)
			dlib.matrix_memcpy_ma(A, A_ptr, dlib.enums.CblasRowMajor)
			dlib.matrix_memcpy_ma(C, C_ptr, dlib.enums.CblasRowMajor)

			a2c = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(a2c, m, k) )
			self.register_var('a2c', a2c, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			a2c_py = np.zeros(m).astype(c_size_t)
			a2c_ptr = a2c_py.ctypes.data_as(dlib.c_size_t_p)

			a2c_py += self.u_test
			dlib.indvector_memcpy_va(a2c.vec, a2c_ptr, 1)

			counts = dlib.vector()
			dlib.vector_calloc(counts, k)
			self.register_var('counts', counts, dlib.vector_free)

			kvec = dlib.vector()
			dlib.vector_calloc(kvec, k)
			self.register_var('kvec', kvec, dlib.vector_free)

			nvec = dlib.vector()
			dlib.vector_calloc(nvec, n)
			self.register_var('nvec', nvec, dlib.vector_free)

			kvec_py = np.zeros(k).astype(dlib.pyfloat)
			kvec_ptr = kvec_py.ctypes.data_as(dlib.ok_float_p)

			nvec_py = np.random.rand(n).astype(dlib.pyfloat)
			nvec_ptr = nvec_py.ctypes.data_as(dlib.ok_float_p)

			h = lib.cluster_aid_p()

			MAXDIST = 10.
			CHANGE_TOL = int(1 + 0.01 * m)
			MAXITER = 500
			VERBOSE = 1
			err = lib.k_means(A, C, a2c, counts, h, MAXDIST, CHANGE_TOL,
							  MAXITER, VERBOSE)

			self.assertEqual( err, 0 )

			self.free_var('A')
			self.free_var('C')
			self.free_var('a2c')
			self.free_var('counts')
			self.free_var('nvec')
			self.free_var('kvec')
			self.free_var('hdl')

			self.assertEqual( dlib.ok_device_reset(), 0 )