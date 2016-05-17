import os
import numpy as np
from numpy import ndarray
from ctypes import c_void_p, c_size_t, byref, cast
from optkit.libs.clustering import ClusteringLibs
from optkit.tests.C.base import OptkitCTestCase

class ClusterLibsTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ClusteringLibs()
		# self.A_test = self.A_test_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.k = int(self.shape[0]**0.5)

		self.C_test = np.random.rand(self.k, self.shape[1])
		self.A_test = np.random.rand(*self.shape)

		# construct A_test as a noisy upsampled version of C
		percent_noise = 0.05
		for i in xrange(self.shape[0]):
			self.A_test[i, :] *= percent_noise
			self.A_test[i, :] += self.C_test[i % self.k, :]

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
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

	def gen_py_upsamplingvec(self, lib, size1, size2, random=False):
		if 'c_size_t_p' not in lib.__dict__:
			raise ValueError('symbol "c_size_t_p" undefined in '
							 'library {}'.format(lib))

		u_py = np.zeros(size1).astype(c_size_t)
		u_ptr = u_py.ctypes.data_as(lib.c_size_t_p)
		if random:
			u_py += (size2 * np.random.rand(size1)).astype(c_size_t)
			u_py[-1] = size2 - 1
		return u_py, u_ptr

	def register_upsamplingvec(self, lib, size1, size2, name, random=False):
		if not 'upsamplingvec_alloc' in lib.__dict__:
			raise ValueError('library {} cannot allocate an '
							 'upsamplingvec'.format(lib))

		u = lib.upsamplingvec()
		self.assertCall( lib.upsamplingvec_alloc(u, size1, size2) )
		self.register_var(name, u, lib.upsamplingvec_free)
		u_py, u_ptr = self.gen_py_upsamplingvec(lib, size1, size2, random)
		if random:
			self.assertCall( lib.indvector_memcpy_va(u.vec, u_ptr, 1) )
		return u, u_py, u_ptr

	def register_cluster_aid(self, lib, n_vectors, n_clusters, order, name):
		if not 'cluster_aid_alloc' in lib.__dict__:
			raise ValueError('library {} cannot allocate a cluster '
							 'aid'.format(lib))

		h = lib.cluster_aid()
		self.assertCall( lib.cluster_aid_alloc(h, n_vectors, n_clusters, order) )
		self.register_var('h', h, lib.cluster_aid_free)
		return h

	def test_upsamplingvec_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			u = lib.upsamplingvec()

			self.assertCall( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)

			self.assertEqual( u.size1, m )
			self.assertEqual( u.size2, k )

			self.assertCall( lib.upsamplingvec_free(u) )
			self.unregister_var('u')
			self.assertCall( lib.ok_device_reset() )

	def test_upsamplingvec_check_bounds(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			u, u_py, u_ptr = self.register_upsamplingvec(lib, m, k, 'u',
														 random=True)

			self.assertCall( lib.upsamplingvec_check_bounds(u) )
			u_py += 2 * k
			self.assertCall( lib.indvector_memcpy_va(u.vec, u_ptr, 1) )

			# expect error
			print '\nexpect dimension mismatch error:'
			err = lib.upsamplingvec_check_bounds(u)
			self.assertEqual( err, lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )

			self.assertCall( lib.upsamplingvec_free(u) )
			self.unregister_var('u')
			self.assertCall( lib.ok_device_reset() )

	def test_upsamplingvec_update_size(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			u, u_py, u_ptr = self.register_upsamplingvec(lib, m, k, 'u',
														 random=True)
			self.assertCall( lib.upsamplingvec_check_bounds(u) )

			# incorrect size
			print '\nexpect dimension mismatch error:'
			u.size2 = k / 2
			err = lib.upsamplingvec_check_bounds(u)
			self.assertEqual( err, lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )

			self.assertCall( lib.upsamplingvec_update_size(u) )
			self.assertEqual( u.size2, k )
			self.assertCall( lib.upsamplingvec_check_bounds(u) )

			self.assertCall( lib.upsamplingvec_free(u) )
			self.unregister_var('u')
			self.assertCall( lib.ok_device_reset() )

	def test_upsamplingvec_subvector(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			offset = m / 4
			msub = m / 2
			ksub = k / 2

			u, u_py, u_ptr = self.register_upsamplingvec(lib, m, k, 'u',
														 random=True)
			usub = lib.upsamplingvec()
			usub_py, usub_ptr = self.gen_py_upsamplingvec(lib, msub, k)

			self.assertCall( lib.upsamplingvec_subvector(usub, u, offset, msub,
									k) )
			self.assertCall( lib.indvector_memcpy_av(usub_ptr, usub.vec, 1) )
			self.assertTrue( usub.size1 == msub )
			self.assertTrue( usub.size2 == k )
			self.assertTrue( sum(usub_py - u_py[offset : offset + msub]) == 0 )

			self.assertCall( lib.upsamplingvec_subvector(usub, u, offset, msub,
									ksub) )
			self.assertTrue( usub.size1 == msub )
			self.assertTrue( usub.size2 == ksub )

			self.assertCall( lib.upsamplingvec_free(u) )
			self.unregister_var('u')
			self.assertCall( lib.ok_device_reset() )

	def test_upsamplingvec_mul_matrix(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)
			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5
			ATOLK = RTOL * k**0.5

			rowmajor = lib.enums.CblasRowMajor
			hdl = self.register_blas_handle(lib, 'hdl')
			A, A_py, A_ptr = self.register_matrix(lib, m, n, rowmajor, 'A',
												  random=True)
			B, B_py, B_ptr = self.register_matrix(lib, n, m, rowmajor, 'B',
												  random=True)
			C, C_py, C_ptr = self.register_matrix(lib, k, n, rowmajor, 'C',
												  random=True)
			D, D_py, D_ptr = self.register_matrix(lib, n, k, rowmajor, 'D',
												  random=True)
			E = lib.matrix(0, 0, 0, None, 0)

			mvec, mvec_py, mvec_ptr = self.register_vector(lib, m, 'mvec',
														   random=True)
			nvec, nvec_py, nvec_ptr = self.register_vector(lib, n, 'nvec',
														   random=True)
			kvec, kvec_py, kvec_ptr = self.register_vector(lib, k, 'kvec',
														   random=True)

			u, u_py, u_ptr = self.register_upsamplingvec(lib, m, k, 'u',
														 random=True)

			T = lib.enums.CblasTrans
			N = lib.enums.CblasNoTrans
			alpha = np.random.rand()
			beta = np.random.rand()

			# functioning cases
			self.upsamplingvec_mul(
					'N', 'N', 'N', alpha, u_py, C_py, beta, A_py)
			result_py = A_py.dot(nvec_py)
			self.assertCall(lib.upsamplingvec_mul_matrix(
					hdl, N, N, N, alpha, u, C, beta, A) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, A, nvec, 0, mvec) )
			self.assertCall( lib.vector_memcpy_av(mvec_ptr, mvec, 1) )
			self.assertVecEqual( mvec_py, result_py, ATOLM, RTOL )

			self.upsamplingvec_mul(
					'T', 'N', 'N', alpha, u_py, A_py, beta, C_py)
			result_py = C_py.dot(nvec_py)
			self.assertCall( lib.upsamplingvec_mul_matrix(
					hdl, T, N, N, alpha, u, A, beta, C) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, C, nvec, 0, kvec) )
			self.assertCall( lib.vector_memcpy_av(kvec_ptr, kvec, 1) )
			self.assertVecEqual( kvec_py, result_py, ATOLK, RTOL )

			self.upsamplingvec_mul(
					'N', 'N', 'T', alpha, u_py, C_py, beta, B_py)
			result_py = B_py.dot(mvec_py)
			self.assertCall( lib.upsamplingvec_mul_matrix(
					hdl, N, N, T, alpha, u, C, beta, B) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, B, mvec, 0, nvec) )
			self.assertCall( lib.vector_memcpy_av(nvec_ptr, nvec, 1) )
			self.assertVecEqual( nvec_py, result_py, ATOLN, RTOL )

			self.upsamplingvec_mul(
					'T', 'N', 'T', alpha, u_py, A_py, beta, D_py)
			result_py = D_py.dot(kvec_py)
			self.assertCall( lib.upsamplingvec_mul_matrix(
					hdl, T, N, T, alpha, u, A, beta, D) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, D, kvec, 0, nvec) )
			self.assertCall( lib.vector_memcpy_av(nvec_ptr, nvec, 1) )
			self.assertVecEqual( nvec_py, result_py, ATOLN, RTOL )

			self.upsamplingvec_mul(
					'N', 'T', 'N', alpha, u_py, D_py, beta, A_py)
			result_py = A_py.dot(nvec_py)
			self.assertCall( lib.upsamplingvec_mul_matrix(
					hdl, N, T, N, alpha, u, D, beta, A) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, A, nvec, 0, mvec) )
			self.assertCall( lib.vector_memcpy_av(mvec_ptr, mvec, 1) )
			self.assertVecEqual( mvec_py, result_py, ATOLM, RTOL )

			self.upsamplingvec_mul(
					'T', 'T', 'N', alpha, u_py, B_py, beta, C_py)
			result_py = C_py.dot(nvec_py)
			self.assertCall( lib.upsamplingvec_mul_matrix(
					hdl, T, T, N, alpha, u, B, beta, C) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, C, nvec, 0, kvec) )
			self.assertCall( lib.vector_memcpy_av(kvec_ptr, kvec, 1) )
			self.assertVecEqual( kvec_py, result_py, ATOLK, RTOL)

			self.upsamplingvec_mul(
					'N', 'T', 'T', alpha, u_py, D_py, beta, B_py)
			result_py = B_py.dot(mvec_py)
			self.assertCall( lib.upsamplingvec_mul_matrix(
					hdl, N, T, T, alpha, u, D, beta, B) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, B, mvec, 0, nvec) )
			self.assertCall( lib.vector_memcpy_av(nvec_ptr, nvec, 1) )
			self.assertVecEqual( nvec_py, result_py, ATOLN, RTOL)

			self.upsamplingvec_mul(
					'T', 'T', 'T', alpha, u_py, B_py, beta, D_py)
			result_py = D_py.dot(kvec_py)
			self.assertCall( lib.upsamplingvec_mul_matrix(
					hdl, T, T, T, alpha, u, B, beta, D) )
			self.assertCall( lib.blas_gemv(hdl, N, 1, D, kvec, 0, nvec) )
			self.assertCall( lib.vector_memcpy_av(nvec_ptr, nvec, 1) )
			self.assertVecEqual( nvec_py, result_py, ATOLN, RTOL)

			# reject: dimension mismatch
			print '\nexpect dimension mismatch error:'
			err = lib.upsamplingvec_mul_matrix(
					hdl, N, N, N, alpha, u, C, beta, B)
			self.assertEqual( err, lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )
			print '\nexpect dimension mismatch error:'
			err = lib.upsamplingvec_mul_matrix(
					hdl, T, N, N, alpha, u, A, beta, D)
			self.assertEqual( err, lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )

			# reject: unallocated
			print '\nexpect unallocated error:'
			err = lib.upsamplingvec_mul_matrix(
					hdl, T, N, N, alpha, u, E, beta, C)
			self.assertEqual( err, lib.enums.OPTKIT_ERROR_UNALLOCATED )

			self.free_vars('A', 'B', 'C', 'D', 'mvec', 'nvec', 'kvec', 'u')
			self.free_var('hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_upsamplingvec_count(self):
		m, n = self.shape
		k = self.k
		hdl = c_void_p()

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			u, u_py, u_ptr = self.register_upsamplingvec(lib, m, k, 'u',
														 random=True)
			counts, counts_py, counts_ptr = self.register_vector(lib, k,
																 'counts')

			self.assertCall( lib.upsamplingvec_count(u, counts) )
			self.assertCall( lib.vector_memcpy_av(counts_ptr, counts, 1) )

			counts_host = np.zeros(k)
			for idx in u_py:
				counts_host[idx] += 1

			self.assertTrue( sum(counts_py - counts_host) < 1e-7 * k**0.5 )

			self.free_vars('u', 'counts')
			self.assertCall( lib.ok_device_reset() )

	def test_cluster_aid_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			h = lib.cluster_aid()

			self.assertCall( lib.cluster_aid_alloc(h, m, k,
							 		lib.enums.CblasRowMajor) )
			self.register_var('h', h, lib.cluster_aid_free)

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

			self.assertCall( lib.cluster_aid_free(h) )
			self.unregister_var('h')

			self.assertEqual( h.a2c_tentative_full.size1, 0 )
			self.assertEqual( h.a2c_tentative_full.size2, 0 )
			self.assertEqual( h.d_min_full.size, 0 )
			self.assertEqual( h.c_squared_full.size, 0 )
			self.assertEqual( h.D_full.size1, 0 )
			self.assertEqual( h.D_full.size2, 0 )
			self.assertCall( lib.ok_device_reset() )

	def test_cluster_aid_resize(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

	def test_kmeans_work_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			w = lib.kmeans_work()
			self.assertCall( lib.kmeans_work_alloc(w, m, k, n) )
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

			self.assertCall( lib.kmeans_work_free(w) )
			self.unregister_var('w')

			self.assertEqual( w.n_vectors, 0 )
			self.assertEqual( w.n_clusters, 0 )
			self.assertEqual( w.vec_length, 0 )
			self.assertCall( lib.ok_device_reset() )

	def test_kmeans_work_resize(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			w = lib.kmeans_work()
			self.assertCall( lib.kmeans_work_alloc(w, m, k, n) )
			self.register_var('w', w, lib.kmeans_work_free)

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

			self.assertCall( lib.kmeans_work_subselect(w, m/2, k/2, n/2) )

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
			self.assertCall( lib.ok_device_reset() )

	def test_kmeans_work_load_extract(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLMN = RTOL * (m * n)**0.5
			ATOLKN = RTOL * (k * n)**0.5
			ATOLM = RTOL * m**0.5
			ATOLK = RTOL * k**0.5

			orderA = orderC = lib.enums.CblasRowMajor
			A, A_ptr = self.gen_py_matrix(lib, m, n, orderA, random=True)
			C, C_ptr = self.gen_py_matrix(lib, k, n, orderC, random=True)
			a2c, a2c_ptr = self.gen_py_upsamplingvec(lib, m, k, random=True)
			counts, counts_ptr = self.gen_py_vector(lib, k)

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
			self.assertCall( lib.kmeans_work_alloc(w, m, k, n) )
			self.register_var('w', w, lib.kmeans_work_free)

			self.assertCall( lib.kmeans_work_load(w, A_ptr, orderA, C_ptr,
												  orderC, a2c_ptr, 1,
												  counts_ptr, 1) )

			lib.matrix_scale(w.C, SCALING)
			lib.vector_scale(w.counts, SCALING)

			self.assertCall( lib.kmeans_work_extract(C_ptr, orderC, a2c_ptr, 1,
													 counts_ptr, 1, w) )

			self.assertVecEqual( A_orig, A, ATOLKN, RTOL )
			self.assertVecEqual( C_orig, C, ATOLKN, RTOL )
			self.assertVecEqual( a2c_orig, a2c, ATOLM, RTOL )
			self.assertVecEqual( counts_orig, counts, ATOLK, RTOL )

			self.free_var('w')
			self.assertCall( lib.ok_device_reset() )

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

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for MAXDIST in [1e3, 0.2]:

				DIGITS = 7 - 2 * single_precision - 1 * gpu
				RTOL = 10**(-DIGITS)
				ATOLM = RTOL * m**0.5
				ATOLN = RTOL * n**0.5
				ATOLK = RTOL * k**0.5

				hdl = self.register_blas_handle(lib, 'hdl')

				orderA = orderC = lib.enums.CblasRowMajor
				A, A_py, A_ptr = self.register_matrix(lib, m, n, orderA, 'A')
				C, C_py, C_ptr = self.register_matrix(lib, k, n, orderC, 'C',
													  random=True)
				a2c, a2c_py, a2c_ptr = self.register_upsamplingvec(
						lib, m, k, 'a2c', random=True)
				a2c_orig = np.zeros(m).astype(c_size_t)
				a2c_orig += a2c_py

				A_py += self.A_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, orderA) )

				D, dmin, reassigned = self.cluster(A_py, C_py, a2c_py, MAXDIST)

				h = self.register_cluster_aid(lib, m, k, orderA, 'h')
				self.assertCall( lib.cluster(A, C, a2c, h, MAXDIST) )

				# compare number of reassignments, C vs Py
				ATOL_REASSIGN = int(1 + 0.1 * m)
				RTOL_REASSIGN = int(1 + 0.1 * reassigned)
				self.assertTrue( abs(h.reassigned - reassigned) <=
								 ATOL_REASSIGN + RTOL_REASSIGN )

				# verify number reassigned
				self.assertCall( lib.indvector_memcpy_av(a2c_ptr, a2c.vec, 1) )
				self.assertEqual( h.reassigned, sum(a2c_py != a2c_orig) )

				# verify all distances
				mvec, mvec_py, mvec_ptr = self.register_vector(lib, m, 'mvec')
				kvec, kvec_py, kvec_ptr = self.register_vector(lib, k, 'kvec')

				mvec_py += np.random.rand(m)

				self.assertCall( lib.vector_memcpy_va(mvec, mvec_ptr, 1) )
				self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1,
							   				   h.D_full, mvec, 0, kvec) )
				self.assertCall( lib.vector_memcpy_av(kvec_ptr, kvec, 1) )

				Dmvec = D.dot(mvec_py)
				self.assertVecEqual( Dmvec, kvec_py, ATOLK, RTOL)

				# verify min distances
				dmin_py = np.zeros(m).astype(lib.pyfloat)
				dmin_ptr = dmin_py.ctypes.data_as(lib.ok_float_p)
				self.assertCall( lib.vector_memcpy_av(
							dmin_ptr, h.d_min_full, 1) )
				self.assertVecEqual( dmin_py, dmin, ATOLM, RTOL )

				self.assertCall( lib.cluster_aid_free(h) )
				self.unregister_var('h')
				self.free_vars('A', 'C', 'a2c', 'mvec', 'kvec', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_calculate_centroids(self):
		""" calculate centroids

			given matrix A, upsampling vector a2c,
			calculate centroid matrix C

			compare C * xrand in Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.register_exit(lib.ok_device_reset)

			hdl = self.register_blas_handle(lib, 'hdl')

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5
			ATOLK = RTOL * k**0.5

			orderA = orderC = lib.enums.CblasRowMajor
			A, A_py, A_ptr = self.register_matrix(lib, m, n, orderA, 'A')
			C, C_py, C_ptr = self.register_matrix(lib, k, n, orderC, 'C',
												  random=True)
			a2c, a2c_py, a2c_ptr = self.register_upsamplingvec(
						lib, m, k, 'a2c', random=True)

			A_py += self.A_test
			self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, orderA) )

			counts, _, _ = self.register_vector(lib, k, 'counts')
			nvec, nvec_py, nvec_ptr = self.register_vector(lib, n, 'nvec')
			kvec, kvec_py, kvec_ptr = self.register_vector(lib, k, 'kvec')

			# C: build centroids
			h = self.register_cluster_aid(lib, m, k, orderA, 'h')
			self.assertCall( lib.calculate_centroids(A, C, a2c, counts, h) )

			# Py: build centroids
			self.upsamplingvec_mul('T', 'N', 'N', 1, a2c_py, A_py, 0, C_py)
			counts_local = np.zeros(k)
			for c in a2c_py:
				counts_local[c] += 1
			for idx, c in enumerate(counts_local):
				counts_local[idx] = (1. / c) if c > 0 else 0.
				C_py[idx, :] *= counts_local[idx]

			# C: kvec = C * nvec
			self.assertCall( lib.vector_memcpy_va(nvec, nvec_ptr, 1) )
			self.assertCall( lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, C,
										   nvec, 0, kvec) )
			self.assertCall( lib.vector_memcpy_av(kvec_ptr, kvec, 1) )

			# Py: kvec = C * nvec
			Cnvec = C_py.dot(nvec_py)

			# compare C vs. Py
			self.assertVecEqual( kvec_py, Cnvec, ATOLK, RTOL)

			self.assertCall( lib.cluster_aid_free(h) )
			self.unregister_var('h')
			self.free_vars('A', 'C', 'a2c', 'counts', 'nvec', 'kvec', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_kmeans(self):
		""" k-means

			given matrix A, cluster # k

			cluster A by kmeans algorithm. compare Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.register_exit(lib.ok_device_reset)

			hdl = self.register_blas_handle(lib, 'hdl')

			orderA = orderC = lib.enums.CblasRowMajor
			A, A_py, A_ptr = self.register_matrix(lib, m, n, orderA, 'A')
			C, C_py, C_ptr = self.register_matrix(lib, k, n, orderC, 'C',
												  random=True)
			a2c, a2c_py, a2c_ptr = self.register_upsamplingvec(
					lib, m, k, 'a2c', random=True)
			counts, _, _ = self.register_vector(lib, k, 'counts')

			A_py += self.A_test
			self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, orderA) )

			DIST_RELTOL = 0.1
			CHANGE_TOL = int(1 + 0.01 * m)
			MAXITER = 500
			VERBOSE = 1

			h = self.register_cluster_aid(lib, m, k, orderA, 'h')
			self.assertCall( lib.k_means(A, C, a2c, counts, h, DIST_RELTOL,
										 CHANGE_TOL, MAXITER, VERBOSE) )

			self.free_vars('h', 'A', 'C', 'a2c', 'counts', 'nvec', 'kvec',
						   'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_kmeans_easy_init_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			w_int = lib.kmeans_easy_init(m, k, n)
			self.assertNotEqual(w_int, 0 )
			work = c_void_p(w_int)
			self.register_var('work', work, lib.kmeans_easy_finish)
			cwork = cast(work, lib.kmeans_work_p)
			self.assertEqual( cwork.contents.n_vectors, m )
			self.assertEqual( cwork.contents.n_clusters, k )
			self.assertEqual( cwork.contents.vec_length, n )
			self.assertCall( lib.kmeans_easy_finish(work) )
			self.unregister_var('work')
			self.assertCall( lib.ok_device_reset() )

	def test_kmeans_easy_resize(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			work = lib.kmeans_easy_init(m, k, n)
			self.register_var('work', work, lib.kmeans_easy_finish)

			self.assertCall( lib.kmeans_easy_resize(work, m/2, k/2, n/2) )

			cwork = cast(work, lib.kmeans_work_p)
			self.assertEqual( cwork.contents.A.size1, m / 2 )
			self.assertEqual( cwork.contents.C.size1, k / 2 )
			self.assertEqual( cwork.contents.A.size2, n / 2 )

			self.free_var('work')
			self.assertCall( lib.ok_device_reset() )

	def test_kmeans_easy_run(self):
		""" k-means (unified call)

			given matrix A, cluster # k

			cluster A by kmeans algorithm. compare Python vs. C
		"""
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.register_exit(lib.ok_device_reset)

			orderA = orderC = lib.enums.CblasRowMajor
			A, A_ptr = self.gen_py_matrix(lib, m, n, orderA)
			C, C_ptr = self.gen_py_matrix(lib, k, n, orderC, random=True)
			a2c, a2c_ptr = self.gen_py_upsamplingvec(lib, m, k, random=True)
			counts, counts_ptr = self.gen_py_vector(lib, k)

			A += self.A_test

			work = lib.kmeans_easy_init(m, k, n)
			self.register_var('work', work, lib.kmeans_easy_finish)

			io = lib.kmeans_io(A_ptr, C_ptr, counts_ptr, a2c_ptr, orderA,
							   orderC, 1, 1)

			DIST_RELTOL = 0.1
			CHANGE_TOL = int(1 + 0.01 * m)
			MAXITER = 500
			VERBOSE = 1
			settings = lib.kmeans_settings(DIST_RELTOL, CHANGE_TOL, MAXITER,
											VERBOSE)

			self.assertCall( lib.kmeans_easy_run(work, settings, io) )

			self.free_var('work')
			self.assertCall( lib.ok_device_reset() )