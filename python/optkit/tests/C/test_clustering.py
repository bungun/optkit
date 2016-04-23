import unittest
import os
import numpy as np
from numpy import ndarray
from ctypes import c_void_p, c_size_t, byref, cast
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
		self.libs = ClusteringLibs()

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
		libs = []
		for (gpu, single_precision) in CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))

		self.assertTrue(any(libs))

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

				dist_inf = np.abs(A[i, :] - C[indmin[i], :]).max()
				if dist_inf <= maxdist:
					a2c[i] = indmin[i]
					reassigned += 1

		return D, dmin, reassigned

	def test_upsamplingvec_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
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
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			u_py = np.zeros(m).astype(c_size_t)
			u_ptr = u_py.ctypes.data_as(lib.c_size_t_p)

			u = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			self.assertEqual( lib.upsamplingvec_check_bounds(u), 0 )
			lib.indvector_memcpy_va(u.vec, u_ptr, 1)
			self.assertEqual( lib.upsamplingvec_check_bounds(u), 0 )
			u_py += 2 * k
			lib.indvector_memcpy_va(u.vec, u_ptr, 1)
			self.assertNotEqual( lib.upsamplingvec_check_bounds(u), 0 )

			err = PRINTERR( lib.upsamplingvec_free(u) )
			self.assertEqual( err, 0 )
			self.unregister_var('u')

	def test_upsamplingvec_update_size(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			u_py = np.random.rand(m).astype(c_size_t)
			u_ptr = u_py.ctypes.data_as(lib.c_size_t_p)
			u_py += self.u_test
			u_py[-1] = k - 1

			u = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			self.assertEqual( lib.upsamplingvec_check_bounds(u), 0 )
			lib.indvector_memcpy_va(u.vec, u_ptr, 1)
			self.assertEqual( lib.upsamplingvec_check_bounds(u), 0 )

			# incorrect size
			u.size2 = k / 2
			self.assertNotEqual( lib.upsamplingvec_check_bounds(u), 0 )

			self.assertEqual( lib.upsamplingvec_update_size(u), 0 )
			self.assertEqual( u.size2, k )
			self.assertEqual( lib.upsamplingvec_check_bounds(u), 0 )

			err = PRINTERR( lib.upsamplingvec_free(u) )
			self.assertEqual( err, 0 )
			self.unregister_var('u')

	def test_upsamplingvec_subvector(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			offset = m / 4
			msub = m / 2
			ksub = k / 2

			u_py = np.zeros(m).astype(c_size_t)
			u_ptr = u_py.ctypes.data_as(lib.c_size_t_p)
			usub_py = np.zeros(m / 2).astype(c_size_t)
			usub_ptr = usub_py.ctypes.data_as(lib.c_size_t_p)

			u = lib.upsamplingvec()
			usub = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			u_py += (k * np.random.rand(m)).astype(c_size_t)
			lib.indvector_memcpy_va(u.vec, u_ptr, 1)
			lib.upsamplingvec_subvector(usub, u, offset, msub, k)
			lib.indvector_memcpy_av(usub_ptr, usub.vec, 1)
			self.assertTrue( usub.size1 == msub )
			self.assertTrue( usub.size2 == k )
			self.assertTrue( sum(usub_py - u_py[offset : offset + msub]) == 0 )

			lib.upsamplingvec_subvector(usub, u, offset, msub, ksub)
			self.assertTrue(usub.size1 == msub)
			self.assertTrue(usub.size2 == ksub)

			err = PRINTERR( lib.upsamplingvec_free(u) )
			self.assertEqual( err, 0 )
			self.unregister_var('u')

	def test_upsamplingvec_mul_matrix(self):
		m, n = self.shape
		k = self.k
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			err = lib.blas_make_handle(byref(hdl))
			self.register_var('hdl', hdl, lib.blas_destroy_handle)
			self.assertEqual( err, 0 )

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5
			ATOLK = RTOL * k**0.5

			A = lib.matrix(0, 0, 0, None, 0)
			B = lib.matrix(0, 0, 0, None, 0)
			C = lib.matrix(0, 0, 0, None, 0)
			D = lib.matrix(0, 0, 0, None, 0)
			E = lib.matrix(0, 0, 0, None, 0)
			lib.matrix_calloc(A, m, n, lib.enums.CblasRowMajor)
			lib.matrix_calloc(B, n, m, lib.enums.CblasRowMajor)
			lib.matrix_calloc(C, k, n, lib.enums.CblasRowMajor)
			lib.matrix_calloc(D, n, k, lib.enums.CblasRowMajor)
			self.register_var('A', A, lib.matrix_free)
			self.register_var('B', B, lib.matrix_free)
			self.register_var('C', C, lib.matrix_free)
			self.register_var('D', D, lib.matrix_free)

			A_py = np.random.rand(m, n).astype(lib.pyfloat)
			B_py = np.random.rand(n, m).astype(lib.pyfloat)
			C_py = np.random.rand(k, n).astype(lib.pyfloat)
			D_py = np.random.rand(n, k).astype(lib.pyfloat)
			A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
			B_ptr = B_py.ctypes.data_as(lib.ok_float_p)
			C_ptr = C_py.ctypes.data_as(lib.ok_float_p)
			D_ptr = D_py.ctypes.data_as(lib.ok_float_p)
			lib.matrix_memcpy_ma(A, A_ptr, lib.enums.CblasRowMajor)
			lib.matrix_memcpy_ma(B, B_ptr, lib.enums.CblasRowMajor)
			lib.matrix_memcpy_ma(C, C_ptr, lib.enums.CblasRowMajor)
			lib.matrix_memcpy_ma(D, D_ptr, lib.enums.CblasRowMajor)

			mvec = lib.vector()
			nvec = lib.vector()
			kvec = lib.vector()
			lib.vector_calloc(mvec, m)
			lib.vector_calloc(nvec, n)
			lib.vector_calloc(kvec, k)
			self.register_var('mvec', mvec, lib.vector_free)
			self.register_var('nvec', nvec, lib.vector_free)
			self.register_var('kvec', kvec, lib.vector_free)

			mvec_py = np.random.rand(m).astype(lib.pyfloat)
			nvec_py = np.random.rand(n).astype(lib.pyfloat)
			kvec_py = np.random.rand(k).astype(lib.pyfloat)
			mvec_ptr = mvec_py.ctypes.data_as(lib.ok_float_p)
			nvec_ptr = nvec_py.ctypes.data_as(lib.ok_float_p)
			kvec_ptr = kvec_py.ctypes.data_as(lib.ok_float_p)
			lib.vector_memcpy_va(mvec, mvec_ptr, 1)
			lib.vector_memcpy_va(nvec, nvec_ptr, 1)
			lib.vector_memcpy_va(kvec, kvec_ptr, 1)

			u = lib.upsamplingvec()
			usub = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			u_py = self.u_test
			u_py[-1] = self.k - 1
			u_ptr = u_py.ctypes.data_as(lib.c_size_t_p)
			lib.indvector_memcpy_va(u.vec, u_ptr, 1)

			T = lib.enums.CblasTrans
			N = lib.enums.CblasNoTrans
			alpha = np.random.rand()
			# beta = np.random.rand()
			beta = 0

			# functioning cases
			self.upsamplingvec_mul('N', 'N', 'N', alpha, u_py, C_py, beta,
				A_py)
			result_py = A_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(N, N, N, alpha, u, C, beta, A)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, A, nvec, 0, mvec)
			lib.vector_memcpy_av(mvec_ptr, mvec, 1)
			self.assertTrue( np.linalg.norm(mvec_py - result_py) <=
							 ATOLM + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'N', 'N', alpha, u_py, A_py, beta,
				C_py)
			result_py = C_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(T, N, N, alpha, u, A, beta, C)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, C, nvec, 0, kvec)
			lib.vector_memcpy_av(kvec_ptr, kvec, 1)
			self.assertTrue( np.linalg.norm(kvec_py - result_py) <=
							 ATOLK + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('N', 'N', 'T', alpha, u_py, C_py, beta,
				B_py)
			result_py = B_py.dot(mvec_py)
			err = lib.upsamplingvec_mul_matrix(N, N, T, alpha, u, C, beta, B)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, B, mvec, 0, nvec)
			lib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'N', 'T', alpha, u_py, A_py, beta, D_py)
			result_py = D_py.dot(kvec_py)
			err = lib.upsamplingvec_mul_matrix(T, N, T, alpha, u, A, beta, D)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, D, kvec, 0, nvec)
			lib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('N', 'T', 'N', alpha, u_py, D_py, beta,
				A_py)
			result_py = A_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(N, T, N, alpha, u, D, beta, A)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, A, nvec, 0, mvec)
			lib.vector_memcpy_av(mvec_ptr, mvec, 1)
			self.assertTrue( np.linalg.norm(mvec_py - result_py) <=
							 ATOLM + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'T', 'N', alpha, u_py, B_py, beta,
				C_py)
			result_py = C_py.dot(nvec_py)
			err = lib.upsamplingvec_mul_matrix(T, T, N, alpha, u, B, beta, C)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, C, nvec, 0, kvec)
			lib.vector_memcpy_av(kvec_ptr, kvec, 1)
			self.assertTrue( np.linalg.norm(kvec_py - result_py) <=
							 ATOLK + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('N', 'T', 'T', alpha, u_py, D_py, beta, B_py)
			result_py = B_py.dot(mvec_py)
			err = lib.upsamplingvec_mul_matrix(N, T, T, alpha, u, D, beta, B)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, B, mvec, 0, nvec)
			lib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			self.upsamplingvec_mul('T', 'T', 'T', alpha, u_py, B_py, beta, D_py)
			result_py = D_py.dot(kvec_py)
			err = lib.upsamplingvec_mul_matrix(T, T, T, alpha, u, B, beta, D)
			self.assertEqual( PRINTERR(err), 0 )
			lib.blas_gemv(hdl, N, 1, D, kvec, 0, nvec)
			lib.vector_memcpy_av(nvec_ptr, nvec, 1)
			self.assertTrue( np.linalg.norm(nvec_py - result_py) <=
							 ATOLN + RTOL * np.linalg.norm(result_py))

			# reject: dimension mismatch
			err = lib.upsamplingvec_mul_matrix(N, N, N, alpha, u, C, beta, B)
			self.assertEqual( err, 11L )
			err = lib.upsamplingvec_mul_matrix(T, N, N, alpha, u, A, beta, D)
			self.assertEqual( err, 11L )

			# reject: unallocated
			err = lib.upsamplingvec_mul_matrix(T, N, N, alpha, u, E, beta, C)
			self.assertEqual( err, 101L )

			self.free_var('A')
			self.free_var('B')
			self.free_var('C')
			self.free_var('D')
			self.free_var('mvec')
			self.free_var('nvec')
			self.free_var('kvec')
			self.free_var('u')
			self.free_var('hdl')

			self.assertEqual( lib.ok_device_reset(), 0 )

	def test_upsamplingvec_count(self):
		m, n = self.shape
		k = self.k
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			u = lib.upsamplingvec()
			usub = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			counts = lib.vector()
			lib.vector_calloc(counts, k)
			self.register_var('counts', counts, lib.vector_free)
			counts_py = np.zeros(k).astype(lib.pyfloat)
			counts_ptr = counts_py.ctypes.data_as(lib.ok_float_p)

			u_py = self.u_test
			u_py[-1] = self.k - 1
			u_ptr = u_py.ctypes.data_as(lib.c_size_t_p)
			lib.indvector_memcpy_va(u.vec, u_ptr, 1)

			lib.upsamplingvec_count(u, counts)
			lib.vector_memcpy_av(counts_ptr, counts, 1)

			counts_host = np.zeros(k)
			for idx in u_py:
				counts_host[idx] += 1

			self.assertTrue( sum(counts_py - counts_host) < 1e-7 * k**0.5 )

			self.free_var('u')
			self.free_var('counts')

	def test_cluster_aid_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			h = lib.cluster_aid()

			err = PRINTERR( lib.cluster_aid_alloc(h, m, k,
				lib.enums.CblasRowMajor) )
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

	# def test_cluster_aid_resize(self):
	# 	m, n = self.shape
	# 	k = self.k

	# 	for (gpu, single_precision) in CONDITIONS:
	# 		lib = self.libs.get(single_precision=single_precision, gpu=gpu)
	# 		if lib is None:
	# 			continue

	def test_kmeans_work_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			w = lib.kmeans_work()
			self.assertEqual( lib.kmeans_work_alloc(w, m, k, n), 0 )
			self.register_var('w', w, lib.kmeans_work_free)

			self.assertNotEqual( w.indicator, 0 )
			self.assertEqual( w.n_vectors, m )
			self.assertEqual( w.n_clusters, k )
			self.assertEqual( w.vec_length, n )
			self.assertEqual( w.A.size1, m )
			self.assertEqual( w.A.size2, n )
			self.assertNotEqual( w.A.data, 0 )
			self.assertEqual( w.C.size1, k )
			self.assertEqual( w.C.size2, n )
			self.assertNotEqual( w.C.data, 0 )
			self.assertEqual( w.counts.size, k )
			self.assertNotEqual( w.counts.data, 0 )
			self.assertEqual( w.a2c.size1, m )
			self.assertEqual( w.a2c.size2, k )
			self.assertNotEqual( w.a2c.indices, 0 )
			self.assertNotEqual( w.h.indicator, 0 )

			self.assertEqual( lib.kmeans_work_free(w), 0)
			self.unregister_var('w')

			self.assertEqual( w.n_vectors, 0 )
			self.assertEqual( w.n_clusters, 0 )
			self.assertEqual( w.vec_length, 0 )



	def test_kmeans_work_resize(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			w = lib.kmeans_work()
			err = PRINTERR( lib.kmeans_work_alloc(w, m, k, n) )
			self.register_var('w', w, lib.kmeans_work_free)
			self.assertEqual( err, 0 )

			self.assertEqual( w.n_vectors, m )
			self.assertEqual( w.n_clusters, k )
			self.assertEqual( w.vec_length, n )
			self.assertEqual( w.A.size1, m )
			self.assertEqual( w.A.size2, n )
			self.assertEqual( w.C.size1, k )
			self.assertEqual( w.C.size2, n )
			self.assertEqual( w.counts.size, k )
			self.assertEqual( w.a2c.size1, m )
			self.assertEqual( w.a2c.size2, k )

			err = PRINTERR( lib.kmeans_work_subselect(
					w, m / 2, k / 2, n / 2) )
			self.assertEqual( err, 0 )

			self.assertEqual( w.n_vectors, m )
			self.assertEqual( w.n_clusters, k )
			self.assertEqual( w.vec_length, n )
			self.assertEqual( w.A.size1, m / 2 )
			self.assertEqual( w.A.size2, n / 2 )
			self.assertEqual( w.C.size1, k / 2 )
			self.assertEqual( w.C.size2, n / 2 )
			self.assertEqual( w.counts.size, k / 2 )
			self.assertEqual( w.a2c.size1, m / 2 )
			self.assertEqual( w.a2c.size2, k / 2 )

			self.free_var('w')

	def test_kmeans_work_load_extract(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLMN = RTOL * (m * n)**0.5
			ATOLKN = RTOL * (k * n)**0.5
			ATOLM = RTOL * m**0.5
			ATOLK = RTOL * k**0.5

			A = np.random.rand(m, n).astype(lib.pyfloat)
			C = np.random.rand(k, n).astype(lib.pyfloat)
			A_ptr = A.ctypes.data_as(lib.ok_float_p)
			C_ptr = C.ctypes.data_as(lib.ok_float_p)
			orderA = lib.enums.CblasRowMajor if A.flags.c_contiguous else \
					 lib.enums.CblasColMajor
			orderC = lib.enums.CblasRowMajor if C.flags.c_contiguous else \
					 lib.enums.CblasColMajor

			a2c = np.zeros(m).astype(c_size_t)
			a2c_ptr = a2c.ctypes.data_as(lib.c_size_t_p)
			a2c += self.u_test

			counts = np.random.rand(k).astype(lib.pyfloat)
			counts_ptr = counts.ctypes.data_as(lib.ok_float_p)

			A_orig = np.zeros((m, n))
			A_orig += A
			C_orig = np.zeros((k, n))
			C_orig += C
			a2c_orig = np.zeros(m)
			a2c_orig += a2c
			counts_orig = np.zeros(k)
			counts_orig += counts

			SCALING = np.random.rand()
			C_orig *= SCALING
			counts_orig *= SCALING

			w = lib.kmeans_work()
			self.assertEqual( lib.kmeans_work_alloc(w, m, k, n), 0 )
			self.register_var('w', w, lib.kmeans_work_free)

			self.assertEqual(
					lib.kmeans_work_load(
							w, A_ptr, orderA, C_ptr, orderC, a2c_ptr, 1,
							counts_ptr, 1), 0 )

			lib.matrix_scale(w.C, SCALING)
			lib.vector_scale(w.counts, SCALING)

			self.assertEqual(
					lib.kmeans_work_extract(
							C_ptr, orderC, a2c_ptr, 1, counts_ptr, 1, w), 0 )

			self.assertTrue( np.linalg.norm(A_orig - A) <=
							 ATOLKN + RTOL * np.linalg.norm(A) )
			self.assertTrue( np.linalg.norm(C_orig - C) <=
							 ATOLKN + RTOL * np.linalg.norm(C) )
			self.assertTrue( np.linalg.norm(a2c_orig - a2c) <=
							 ATOLM + RTOL * np.linalg.norm(a2c) )
			self.assertTrue( np.linalg.norm(counts_orig - counts) <=
							 ATOLK + RTOL * np.linalg.norm(counts) )

			self.free_var('w')

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
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for MAXDIST in [1e3, 0.2]:

				hdl = c_void_p()
				err = lib.blas_make_handle(byref(hdl))
				self.register_var('hdl', hdl, lib.blas_destroy_handle)
				self.assertEqual( err, 0 )

				DIGITS = 7 - 2 * single_precision - 1 * gpu
				RTOL = 10**(-DIGITS)
				ATOLM = RTOL * m**0.5
				ATOLN = RTOL * n**0.5
				ATOLK = RTOL * k**0.5

				A = lib.matrix(0, 0, 0, None, 0)
				C = lib.matrix(0, 0, 0, None, 0)
				lib.matrix_calloc(A, m, n, lib.enums.CblasRowMajor)
				lib.matrix_calloc(C, k, n, lib.enums.CblasRowMajor)
				self.register_var('A', A, lib.matrix_free)
				self.register_var('C', C, lib.matrix_free)

				A_py = np.random.rand(m, n).astype(lib.pyfloat)
				C_py = np.random.rand(k, n).astype(lib.pyfloat)
				A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
				C_ptr = C_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_memcpy_ma(A, A_ptr, lib.enums.CblasRowMajor)
				lib.matrix_memcpy_ma(C, C_ptr, lib.enums.CblasRowMajor)

				a2c = lib.upsamplingvec()
				err = PRINTERR( lib.upsamplingvec_alloc(a2c, m, k) )
				self.register_var('a2c', a2c, lib.upsamplingvec_free)
				self.assertEqual( err, 0 )

				a2c_py = np.zeros(m).astype(c_size_t)
				a2c_ptr = a2c_py.ctypes.data_as(lib.c_size_t_p)

				a2c_py += self.u_test
				lib.indvector_memcpy_va(a2c.vec, a2c_ptr, 1)

				h = lib.cluster_aid_p()

				D, dmin, reassigned = self.cluster(A_py, C_py, a2c_py, MAXDIST)

				err = PRINTERR( lib.cluster(A, C, a2c, byref(h), MAXDIST) )
				self.register_var('h', h, lib.cluster_aid_free)
				self.assertEqual( err, 0 )

				# compare number of reassignments, C vs Py
				ATOL_REASSIGN = int(1 + 0.1 * k)
				RTOL_REASSIGN = int(1 + 0.1 * reassigned)
				self.assertTrue( abs(h.contents.reassigned - reassigned) <=
								 ATOL_REASSIGN + RTOL_REASSIGN )

				# verify number reassigned
				lib.indvector_memcpy_av(a2c_ptr, a2c.vec, 1)
				self.assertEqual( h.contents.reassigned,
					sum(a2c_py != self.u_test) )

				# verify all distances
				kvec = lib.vector()
				lib.vector_calloc(kvec, k)
				self.register_var('kvec', kvec, lib.vector_free)

				mvec = lib.vector()
				lib.vector_calloc(mvec, m)
				self.register_var('mvec', mvec, lib.vector_free)

				kvec_py = np.zeros(k).astype(lib.pyfloat)
				kvec_ptr = kvec_py.ctypes.data_as(lib.ok_float_p)

				mvec_py = np.random.rand(m).astype(lib.pyfloat)
				mvec_ptr = mvec_py.ctypes.data_as(lib.ok_float_p)

				lib.vector_memcpy_va(mvec, mvec_ptr, 1)
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1,
							   h.contents.D_full, mvec, 0, kvec)
				lib.vector_memcpy_av(kvec_ptr, kvec, 1)

				Dmvec = D.dot(mvec_py)
				self.assertTrue( np.linalg.norm(Dmvec - kvec_py) <=
								 ATOLK + RTOL * np.linalg.norm(Dmvec) )

				# verify min distances
				dmin_py = np.zeros(m).astype(lib.pyfloat)
				dmin_ptr = dmin_py.ctypes.data_as(lib.ok_float_p)
				lib.vector_memcpy_av(dmin_ptr, h.contents.d_min_full, 1)
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

				self.assertEqual( lib.ok_device_reset(), 0 )

	def test_calculate_centroids(self):
		""" calculate centroids

			given matrix A, upsampling vector a2c,
			calculate centroid matrix C

			compare C * xrand in Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			hdl = c_void_p()
			err = lib.blas_make_handle(byref(hdl))
			self.register_var('hdl', hdl, lib.blas_destroy_handle)
			self.assertEqual( err, 0 )

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5
			ATOLK = RTOL * k**0.5

			A = lib.matrix(0, 0, 0, None, 0)
			C = lib.matrix(0, 0, 0, None, 0)
			lib.matrix_calloc(A, m, n, lib.enums.CblasRowMajor)
			lib.matrix_calloc(C, k, n, lib.enums.CblasRowMajor)
			self.register_var('A', A, lib.matrix_free)
			self.register_var('C', C, lib.matrix_free)

			A_py = np.random.rand(m, n).astype(lib.pyfloat)
			C_py = np.random.rand(k, n).astype(lib.pyfloat)
			A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
			C_ptr = C_py.ctypes.data_as(lib.ok_float_p)
			lib.matrix_memcpy_ma(A, A_ptr, lib.enums.CblasRowMajor)
			lib.matrix_memcpy_ma(C, C_ptr, lib.enums.CblasRowMajor)

			a2c = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(a2c, m, k) )
			self.register_var('a2c', a2c, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			a2c_py = np.zeros(m).astype(c_size_t)
			a2c_ptr = a2c_py.ctypes.data_as(lib.c_size_t_p)

			a2c_py += self.u_test
			lib.indvector_memcpy_va(a2c.vec, a2c_ptr, 1)

			counts = lib.vector()
			lib.vector_calloc(counts, k)
			self.register_var('counts', counts, lib.vector_free)

			kvec = lib.vector()
			lib.vector_calloc(kvec, k)
			self.register_var('kvec', kvec, lib.vector_free)

			nvec = lib.vector()
			lib.vector_calloc(nvec, n)
			self.register_var('nvec', nvec, lib.vector_free)

			kvec_py = np.zeros(k).astype(lib.pyfloat)
			kvec_ptr = kvec_py.ctypes.data_as(lib.ok_float_p)

			nvec_py = np.random.rand(n).astype(lib.pyfloat)
			nvec_ptr = nvec_py.ctypes.data_as(lib.ok_float_p)

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
			lib.vector_memcpy_va(nvec, nvec_ptr, 1)
			lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, C, nvec, 0,
						   kvec)
			lib.vector_memcpy_av(kvec_ptr, kvec, 1)

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

			self.assertEqual( lib.ok_device_reset(), 0 )

	def test_kmeans(self):
		""" k-means

			given matrix A, cluster # k

			cluster A by kmeans algorithm. compare Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			hdl = c_void_p()
			err = lib.blas_make_handle(byref(hdl))
			self.register_var('hdl', hdl, lib.blas_destroy_handle)
			self.assertEqual( err, 0 )

			A = lib.matrix(0, 0, 0, None, 0)
			C = lib.matrix(0, 0, 0, None, 0)
			lib.matrix_calloc(A, m, n, lib.enums.CblasRowMajor)
			lib.matrix_calloc(C, k, n, lib.enums.CblasRowMajor)
			self.register_var('A', A, lib.matrix_free)
			self.register_var('C', C, lib.matrix_free)

			A_py = np.random.rand(m, n).astype(lib.pyfloat)
			C_py = np.random.rand(k, n).astype(lib.pyfloat)
			A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
			C_ptr = C_py.ctypes.data_as(lib.ok_float_p)
			lib.matrix_memcpy_ma(A, A_ptr, lib.enums.CblasRowMajor)
			lib.matrix_memcpy_ma(C, C_ptr, lib.enums.CblasRowMajor)

			a2c = lib.upsamplingvec()
			err = PRINTERR( lib.upsamplingvec_alloc(a2c, m, k) )
			self.register_var('a2c', a2c, lib.upsamplingvec_free)
			self.assertEqual( err, 0 )

			a2c_py = np.zeros(m).astype(c_size_t)
			a2c_ptr = a2c_py.ctypes.data_as(lib.c_size_t_p)

			a2c_py += self.u_test
			lib.indvector_memcpy_va(a2c.vec, a2c_ptr, 1)

			counts = lib.vector()
			lib.vector_calloc(counts, k)
			self.register_var('counts', counts, lib.vector_free)

			kvec = lib.vector()
			lib.vector_calloc(kvec, k)
			self.register_var('kvec', kvec, lib.vector_free)

			nvec = lib.vector()
			lib.vector_calloc(nvec, n)
			self.register_var('nvec', nvec, lib.vector_free)

			kvec_py = np.zeros(k).astype(lib.pyfloat)
			kvec_ptr = kvec_py.ctypes.data_as(lib.ok_float_p)

			nvec_py = np.random.rand(n).astype(lib.pyfloat)
			nvec_ptr = nvec_py.ctypes.data_as(lib.ok_float_p)

			h = lib.cluster_aid_p()

			DIST_RELTOL = 0.1
			CHANGE_TOL = int(1 + 0.01 * m)
			MAXITER = 500
			VERBOSE = 1
			err = lib.k_means(A, C, a2c, counts, h, DIST_RELTOL, CHANGE_TOL,
							  MAXITER, VERBOSE)

			self.assertEqual( err, 0 )

			self.free_var('A')
			self.free_var('C')
			self.free_var('a2c')
			self.free_var('counts')
			self.free_var('nvec')
			self.free_var('kvec')
			self.free_var('hdl')

			self.assertEqual( lib.ok_device_reset(), 0 )

	def test_kmeans_easy_init_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			work = lib.kmeans_easy_init(m, k, n)
			self.register_var('work', work, lib.kmeans_easy_finish)
			self.assertNotEqual( work, 0 )
			cwork = cast(work, lib.kmeans_work_p)
			self.assertEqual(cwork.contents.n_vectors, m)
			self.assertEqual(cwork.contents.n_clusters, k)
			self.assertEqual(cwork.contents.vec_length, n)
			self.assertEqual( lib.kmeans_easy_finish(work), 0 )

	def test_kmeans_easy_resize(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			work = lib.kmeans_easy_init(m, k, n)
			self.register_var('work', work, lib.kmeans_easy_finish)

			err = PRINTERR( lib.kmeans_easy_resize(work, m / 2, k / 2, n / 2) )
			self.assertEqual( err, 0 )

			cwork = cast(work, lib.kmeans_work_p)
			self.assertEqual(cwork.contents.A.size1, m / 2)
			self.assertEqual(cwork.contents.C.size1, k / 2)
			self.assertEqual(cwork.contents.A.size2, n / 2)

			self.free_var('work')

	def test_kmeans_easy_run(self):
		""" k-means

			given matrix A, cluster # k

			cluster A by kmeans algorithm. compare Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			A = np.random.rand(m, n).astype(lib.pyfloat)
			C = np.random.rand(k, n).astype(lib.pyfloat)
			A_ptr = A.ctypes.data_as(lib.ok_float_p)
			C_ptr = C.ctypes.data_as(lib.ok_float_p)
			orderA = lib.enums.CblasRowMajor if A.flags.c_contiguous else \
					 lib.enums.CblasColMajor
			orderC = lib.enums.CblasRowMajor if C.flags.c_contiguous else \
					 lib.enums.CblasColMajor

			a2c = np.zeros(m).astype(c_size_t)
			a2c_ptr = a2c.ctypes.data_as(lib.c_size_t_p)
			a2c += self.u_test

			counts = np.zeros(k).astype(lib.pyfloat)
			counts_ptr = counts.ctypes.data_as(lib.ok_float_p)

			work = lib.kmeans_easy_init(m, k, n)
			self.register_var('work', work, lib.kmeans_easy_finish)

			io = lib.kmeans_io(A_ptr, C_ptr, counts_ptr, a2c_ptr, orderA, orderC, 1, 1)

			DIST_RELTOL = 0.1
			CHANGE_TOL = int(1 + 0.01 * m)
			MAXITER = 500
			VERBOSE = 1
			settings = lib.kmeans_settings(DIST_RELTOL, CHANGE_TOL, MAXITER,
											VERBOSE)

			self.assertEqual( lib.kmeans_easy_run(work, settings, io), 0 )

			self.free_var('work')