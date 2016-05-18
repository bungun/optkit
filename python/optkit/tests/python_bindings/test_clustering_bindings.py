import numpy as np
import gc
import os
from os import path
from ctypes import CDLL, c_size_t
# from optkit import *
from optkit.api import backend
from optkit.types.clustering import *
from optkit.tests.defs import OptkitTestCase

class ClusteringBindingsTestCase(OptkitTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.k = int(self.shape[0]**0.5)
		self.C_test = np.random.rand(self.k, self.shape[1])
		self.A_test = np.random.rand(*self.shape)
		self.assignments_test = (self.k * np.random.rand(self.shape[0])).astype(int)

		# construct A_test as a noisy upsampled version of C
		percent_noise = 0.05
		for i in xrange(self.shape[0]):
			self.A_test[i, :] *= percent_noise
			self.A_test[i, :] += self.C_test[i % self.k, :]

	def test_nearest_triple(self):
		self.assertEqual( nearest_triple(9), (2, 2, 2) )
		self.assertEqual( nearest_triple(10), (2, 3, 2) )
		self.assertEqual( nearest_triple(20), (2, 3, 3) )
		self.assertEqual( nearest_triple(30), (3, 3, 3) )
		self.assertEqual( nearest_triple(100), (4, 5, 5) )
		self.assertEqual( nearest_triple(1000), (10, 10, 10) )

	def test_regcluster(self):
		floor = 20
		floor_block = 2
		size = 150 - floor
		frac = 0.2

		x_full = floor + int(size * np.random.rand())
		y_full = floor + int(size * np.random.rand())
		z_full = floor + int(size * np.random.rand())

		full_size = (x_full, y_full, z_full)

		x_blk = floor_block + int(frac * np.random.rand() * x_full)
		y_blk = floor_block + int(frac * np.random.rand() * y_full)
		z_blk = floor_block + int(frac * np.random.rand() * z_full)

		block_size = (x_blk, y_blk, z_blk)

		assignments = regcluster(block_size, full_size)
		self.assertEqual( len(assignments), x_full * y_full * z_full)

		counts = np.zeros(assignments.max() + 1)
		for a in assignments:
			counts[a] += 1

		# assign at most <block_size> units to a block index
		self.assertTrue( counts.max() <= x_blk * y_blk * z_blk )

	def test_upsampling_vector(self):
		m = 200
		k = 20

		indices = (k * np.random.rand(m)).astype(int)
		u = UpsamplingVector(indices=indices)

		self.assertEqual( u.size1, m )
		self.assertTrue( u.size2 <= k )

		# choose a non-zero assignment
		val = 0
		for i in xrange(m):
			if u.indices[i] > 0:
				val = u.indices[i]
				break


		# number of assignments to 0 + number of assignments to val
		oldsum = u.counts[0] + u.counts[val]
		oldmax = u.max_assignment + 1

		# swap all instances of this non-zero assignment to 0
		for i in xrange(m):
			if u.indices[i] == val:
				u.indices[i] = 0

		# number of assignments to 0: should cover previous 2 bins
		newsum = u.counts[0]
		self.assertEqual( oldsum, newsum )

		# make assignments contiguous (skip no indices)
		u.clean_assignments()
		newmax = u.max_assignment + 1
		self.assertTrue( newmax < oldmax )

	def test_clustering_settings(self):
		ct = ClusteringTypes(backend)
		m = 500

		s = ct.ClusteringSettings(m)
		self.assertTrue( isinstance(backend.cluster, CDLL ))

		self.assertEqual( type(s.pointer), backend.cluster.kmeans_settings )
		self.assertEqual( s.distance_tol, s.DISTANCE_RTOL_DEFAULT )
		self.assertEqual(
				s.assignment_tol, int(np.ceil(m * s.REASSIGN_RTOL_DEFAULT)) )
		self.assertEqual( s.maxiter, s.MAXITER_DEFAULT )
		self.assertEqual( s.verbose, s.VERBOSE_DEFAULT )

		DTOL = 1e-3
		ITOL = 5e-2
		itolm = int(np.ceil(m * ITOL))
		ITER = 200
		VERBOSE = 0

		s = ct.ClusteringSettings(m, DTOL, ITOL, ITER, VERBOSE)
		self.assertEqual( s.distance_tol, DTOL )
		self.assertEqual( s.assignment_tol, itolm )
		self.assertEqual( s.maxiter, ITER )
		self.assertEqual( s.verbose, VERBOSE )

	def test_clustering_work(self):
		ct = ClusteringTypes(backend)
		m = 500
		n = 200
		k = 40

		w = ct.ClusteringWork(m, k, n)

		# check that the C object is tracked by opkit backend module
		self.assertFalse( backend.device_reset_allowed )
		del w
		gc.collect()
		self.assertTrue( backend.device_reset_allowed )

		w = ct.ClusteringWork(m, k, n)

		self.assertEqual( w.m, m )
		self.assertEqual( w.k, k )
		self.assertEqual( w.n, n )
		err = w.resize(m / 2, k / 2)
		self.assertEqual( err, 0 )
		self.assertEqual( w.m, m / 2 )
		self.assertEqual( w.k, k / 2 )
		self.assertEqual( w.n, n )

	def test_clustering_object(self):
		ct = ClusteringTypes(backend)
		m = 500
		n = 200
		k = 40

		clu = ct.Clustering()
		clu.kmeans_work_init(m, k, n)
		self.assertFalse( backend.device_reset_allowed )
		clu.kmeans_work_finish()
		self.assertTrue( backend.device_reset_allowed )

		self.assertTrue( clu.A is None )
		self.assertTrue( clu.A_ptr is None )
		clu.A = np.random.rand(m, n)
		self.assertFalse( clu.A is None )
		self.assertTrue( clu.A.dtype == backend.cluster.pyfloat )
		self.assertTrue( isinstance(clu.A_ptr, backend.cluster.ok_float_p) )

		self.assertTrue( clu.C is None )
		self.assertTrue( clu.C_ptr is None )
		self.assertTrue( clu.counts is None )
		self.assertTrue( clu.counts_ptr is None )
		clu.C = np.random.rand(k, n)
		self.assertFalse( clu.C is None )
		self.assertTrue( clu.C.dtype == backend.cluster.pyfloat )
		self.assertTrue( isinstance(clu.C_ptr, backend.cluster.ok_float_p) )
		self.assertFalse( clu.counts is None )
		self.assertTrue( clu.counts.dtype == backend.cluster.pyfloat )
		self.assertTrue( isinstance(clu.counts_ptr, backend.cluster.ok_float_p) )

		self.assertTrue( clu.a2c is None )
		self.assertTrue( clu.a2c_ptr is None )
		clu.a2c = k * np.random.rand(m)
		self.assertFalse( clu.a2c is None )
		self.assertTrue( clu.a2c.dtype == c_size_t )
		self.assertTrue( isinstance(clu.a2c_ptr, backend.cluster.c_size_t_p) )

		self.assertTrue( isinstance(clu.io, backend.cluster.kmeans_io) )

	def test_kmeans_inplace(self):
		ct = ClusteringTypes(backend)
		clu = ct.Clustering()

		C, a2c, counts = clu.kmeans_inplace(self.A_test, self.C_test,
										 self.assignments_test)
		self.assertTrue( sum(counts) == self.shape[0] )

	def test_kmeans_inplace(self):
		ct = ClusteringTypes(backend)
		clu = ct.Clustering()

		C, a2c, counts = clu.kmeans(self.A_test, self.k,
									   self.assignments_test)
		self.assertTrue( sum(counts) == self.shape[0] )

	def test_blockwise_kmeans_inplace(self):
		ct = ClusteringTypes(backend)
		clu = ct.Clustering()

		b = 3
		A_array = b * [self.A_test]
		C_array = b * [self.C_test]
		a2c_array = b * [self.assignments_test]

		C, a2c, counts = clu.blockwise_kmeans_inplace(A_array, C_array,
													  a2c_array)

		self.assertTrue( len(C) == b )
		self.assertTrue( len(a2c) == b )
		self.assertTrue( len(counts) == b )

		for i in xrange(b):
			self.assertTrue( sum(counts[i]) == self.shape[0] )

	def test_blockwise_kmeans(self):
		ct = ClusteringTypes(backend)
		clu = ct.Clustering()

		b = 3
		A_array = b * [self.A_test]
		a2c_array = b * [self.assignments_test]

		C, a2c, counts = clu.blockwise_kmeans(A_array, a2c_array)

		self.assertTrue( len(C) == b )
		self.assertTrue( len(a2c) == b )
		self.assertTrue( len(counts) == b )

		for i in xrange(b):
			self.assertTrue( sum(counts[i]) == self.shape[0] )


