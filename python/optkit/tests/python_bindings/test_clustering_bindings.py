from optkit.compat import *

import gc
import os
import numpy as np
import ctypes as ct
import unittest

from optkit import api
from optkit.libs import error
from optkit.types import clustering
from optkit.tests import defs
from optkit.tests.python_bindings import prepare

prepare.establish_backend()
backend = api.backend
NO_ERR = error.NO_ERR

class ClusteringBindingsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.shape = defs.shape()
        self.k = int(self.shape[0]**0.5)
        self.C_test = np.random.rand(self.k, self.shape[1])
        self.A_test = np.random.rand(*self.shape)
        self.assignments_test = (self.k * np.random.rand(
                self.shape[0])).astype(int)

        # construct A_test as a noisy upsampled version of C
        percent_noise = 0.05
        for i in xrange(self.shape[0]):
            self.A_test[i, :] *= percent_noise
            self.A_test[i, :] += self.C_test[i % self.k, :]

    def test_nearest_triple(self):
        assert ( clustering.nearest_triple(9) == (2, 2, 2) )
        assert ( clustering.nearest_triple(10) == (2, 3, 2) )
        assert ( clustering.nearest_triple(20) == (2, 3, 3) )
        assert ( clustering.nearest_triple(30) == (3, 3, 3) )
        assert ( clustering.nearest_triple(100) == (4, 5, 5) )
        assert ( clustering.nearest_triple(1000) == (10, 10, 10) )

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

        assignments = clustering.regcluster(block_size, full_size)
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
        u = clustering.UpsamplingVector(indices=indices)

        assert ( u.size1 == m )
        assert ( u.size2 <= k )

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
        clus = clustering.ClusteringTypes(backend)
        m = 500

        s = clus.ClusteringSettings(m)
        self.assertIsInstance( backend.cluster, ct.CDLL )

        assert( type(s.pointer) == backend.cluster.kmeans_settings )
        assert( s.distance_tol == s.DISTANCE_RTOL_DEFAULT )
        assert( s.assignment_tol == int(np.ceil(m * s.REASSIGN_RTOL_DEFAULT)) )
        assert( s.maxiter == s.MAXITER_DEFAULT )
        assert( s.verbose == s.VERBOSE_DEFAULT )

        DTOL = 1e-3
        ITOL = 5e-2
        itolm = int(np.ceil(m * ITOL))
        ITER = 200
        VERBOSE = 0

        s = clus.ClusteringSettings(m, DTOL, ITOL, ITER, VERBOSE)
        assert ( s.distance_tol == DTOL )
        assert ( s.assignment_tol == itolm )
        assert ( s.maxiter == ITER )
        assert ( s.verbose == VERBOSE )

    def test_clustering_work(self):
        clus = clustering.ClusteringTypes(backend)
        m = 500
        n = 200
        k = 40

        w = clus.ClusteringWork(m, k, n)

        # check that the C object is tracked by opkit backend module
        self.assertFalse( backend.device_reset_allowed )
        del w
        gc.collect()
        self.assertTrue( backend.device_reset_allowed )

        w = clus.ClusteringWork(m, k, n)

        assert ( w.m == m )
        assert ( w.k == k )
        assert ( w.n == n )
        assert NO_ERR( w.resize(int(m / 2), int(k / 2)) )
        assert ( w.m == int(m / 2) )
        assert ( w.k == int(k / 2) )
        assert ( w.n == n )

    def test_clustering_object(self):
        clus = clustering.ClusteringTypes(backend)
        m = 500
        n = 200
        k = 40

        clu = clus.Clustering()
        clu.kmeans_work_init(m, k, n)
        assert( not backend.device_reset_allowed )
        clu.kmeans_work_finish()
        assert( backend.device_reset_allowed )

        assert ( clu.A is None )
        assert ( clu.A_ptr is None )
        clu.A = np.random.rand(m, n)
        assert ( clu.A is not None )
        assert ( clu.A.dtype == backend.cluster.pyfloat )
        assert isinstance( clu.A_ptr, backend.cluster.ok_float_p )

        assert ( clu.C is None )
        assert ( clu.C_ptr is None )
        assert ( clu.counts is None)
        assert ( clu.counts_ptr is None )
        clu.C = np.random.rand(k, n)
        assert ( clu.C is not None )
        assert ( clu.C.dtype == backend.cluster.pyfloat )
        assert isinstance( clu.C_ptr, backend.cluster.ok_float_p )
        assert ( clu.counts is not None )
        assert ( clu.counts.dtype == backend.cluster.pyfloat )
        assert isinstance( clu.counts_ptr, backend.cluster.ok_float_p )

        assert ( clu.a2c is None )
        assert ( clu.a2c_ptr is None )
        clu.a2c = k * np.random.rand(m)
        assert ( clu.a2c is not None )
        assert ( clu.a2c.dtype == ct.c_size_t )
        assert isinstance( clu.a2c_ptr, backend.cluster.c_size_t_p )

        assert isinstance( clu.io, backend.cluster.kmeans_io )

    def test_kmeans_inplace(self):
        clus = clustering.ClusteringTypes(backend)
        clu = clus.Clustering()

        C, a2c, counts = clu.kmeans_inplace(self.A_test, self.C_test,
                                         self.assignments_test)
        assert ( sum(counts) == self.shape[0] )

    def test_kmeans_inplace(self):
        clus = clustering.ClusteringTypes(backend)
        clu = clus.Clustering()

        C, a2c, counts = clu.kmeans(self.A_test, self.k,
                                       self.assignments_test)
        assert ( sum(counts) == self.shape[0] )

    def blockwise_outputs_consistent(self, C, a2c, counts, b):
        assert ( len(C) == b )
        assert ( len(a2c) == b )
        assert ( len(counts) == b )
        assert all((sum(counts[i]) == self.shape[0] for i in range(b)))
        return True

    def test_blockwise_kmeans_inplace(self):
        clus = clustering.ClusteringTypes(backend)
        clu = clus.Clustering()

        b = 3
        A_array = b * [self.A_test]
        C_array = b * [self.C_test]
        a2c_array = b * [self.assignments_test]

        C, a2c, counts = clu.blockwise_kmeans_inplace(
                A_array, C_array,a2c_array)
        assert self.blockwise_outputs_consistent(C, a2c, counts, b)

    def test_blockwise_kmeans(self):
        clus = clustering.ClusteringTypes(backend)
        clu = clus.Clustering()

        b = 3
        A_array = b * [self.A_test]
        a2c_array = b * [self.assignments_test]

        C, a2c, counts = clu.blockwise_kmeans(A_array, a2c_array)
        assert self.blockwise_outputs_consistent(C, a2c, counts, b)

