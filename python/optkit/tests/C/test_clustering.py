from optkit.compat import *

import os
import numpy as np
import ctypes as ct
import unittest

from optkit.libs.clustering import ClusteringLibs
from optkit.tests import defs
from optkit.tests.C import context_managers as okcctx
from optkit.tests.C import statements

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances
CUSTOM_TOLS = statements.custom_tolerances

def upsamplingvec_mul(tU, tA, tB, alpha, u, A, beta, B):
    if not bool(isinstance(u, np.ndarray) and isinstance(A, np.ndarray)
                and isinstance(B, np.ndarray)):
        raise TypeError(
                'u, A, and B must be of type {}'.format(np.ndarray))
    if not (len(A.shape) == 2 and len(B.shape) == 2):
        raise ValueError(
                'A and B must be 2-d {}s'.format(np.ndarray))

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

def cluster(A, C, a2c, maxdist):
    if not bool(isinstance(a2c, np.ndarray) and isinstance(A, np.ndarray)
                and isinstance(C, np.ndarray)):
        raise TypeError(
                'a2c, A, and C must be of type {}'.format(np.ndarray))
    if not (len(A.shape) == 2 and len(C.shape) == 2):
        raise ValueError(
                'A and C must be 2-d {}s'.format(np.ndarray))
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

def gen_py_upsamplingvec(lib, size1, size2, random=False):
    if 'c_size_t_p' not in lib.__dict__:
        raise ValueError(
                'symbol "c_size_t_p" undefined in library {}'
                ''.format(lib))

    u_py = np.zeros(size1).astype(ct.c_size_t)
    u_ptr = u_py.ctypes.data_as(lib.c_size_t_p)
    if random:
        u_py += (size2 * np.random.rand(size1)).astype(ct.c_size_t)
        u_py[-1] = size2 - 1
    return u_py, u_ptr

class UpsamplingVecContext(okcctx.CArrayContext, okcctx.CArrayIO):
    def __init__(self, lib, size1, size2, random=False):
        u = lib.upsamplingvec()
        u_py, u_ptr = gen_py_upsamplingvec(lib, size1, size2, random=random)

        def arr2ptr(arr): return np.ravel(arr).ctypes.data_as(type(u_ptr))
        def py2c(py_array):
            return lib.indvector_memcpy_va(u.vec, arr2ptr(py_array), 1)
        def c2py(py_array):
            return lib.indvector_memcpy_av(arr2ptr(py_array), u.vec, 1)
        def build():
            assert NO_ERR( lib.upsamplingvec_alloc(u, size1, size2) )
            if random:
                assert NO_ERR( py2c(u_py) )
        def free(): return lib.upsamplingvec_free(u)

        okcctx.CArrayContext.__init__(self, u, u_py, u_ptr, build, free)
        okcctx.CArrayIO.__init__(self, u_py, py2c, c2py)

class ClusterAidContext(okcctx.CVariableContext):
    def __init__(self, lib, n_vector, n_clusters, order):
        h = lib.cluster_aid()
        def build_():
            assert NO_ERR(lib.cluster_aid_alloc(h, n_vector, n_clusters, order))
            return h
        def free_(): return lib.cluster_aid_free(h)
        okcctx.CVariableContext.__init__(self, build_, free_)

class KmeansWorkContext(okcctx.CVariableContext):
    def __init__(self, lib, points, clusters, size):
        w = lib.kmeans_work()
        def build_():
            assert NO_ERR( lib.kmeans_work_alloc(w, points, clusters, size) )
            return w
        def free_(): return lib.kmeans_work_free(w)
        okcctx.CVariableContext.__init__(self, build_, free_)

class ClusterLibsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(ClusteringLibs())

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.k = int(defs.shape()[0]**0.5)
        self.C_test = np.random.rand(self.k, defs.shape()[1])
        self.A_test = np.random.rand(*defs.shape())

        # construct A_test as a noisy upsampled version of C
        percent_noise = 0.05
        for i in xrange(defs.shape()[0]):
            self.A_test[i, :] *= percent_noise
            self.A_test[i, :] += self.C_test[i % self.k, :]

    def test_libs_exist(self):
        assert any(self.libs)

    def test_upsamplingvec_alloc_free(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                u = lib.upsamplingvec()
                try:
                    assert NO_ERR( lib.upsamplingvec_alloc(u, m, k) )
                    assert ( u.size1 == m )
                    assert ( u.size2 == k )
                finally:
                    assert NO_ERR( lib.upsamplingvec_free(u) )


    def test_upsamplingvec_check_bounds(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                with UpsamplingVecContext(lib, m, k, random=True) as u:
                    assert NO_ERR( lib.upsamplingvec_check_bounds(u.c) )
                    u.py += 2 * k
                    u.sync_to_c()

                    # expect error
                    print('\nexpect dimension mismatch error:')
                    err = lib.upsamplingvec_check_bounds(u.c)
                    assert ( err == lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )


    def test_upsamplingvec_update_size(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                with UpsamplingVecContext(lib, m, k, random=True) as u:
                    assert NO_ERR( lib.upsamplingvec_check_bounds(u.c) )

                    # incorrect size
                    print('\nexpect dimension mismatch error:')
                    u.c.size2 = int(k / 2)
                    err = lib.upsamplingvec_check_bounds(u.c)
                    assert ( err == lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )

                    assert NO_ERR( lib.upsamplingvec_update_size(u.c) )
                    assert ( u.c.size2 == k )
                    assert NO_ERR( lib.upsamplingvec_check_bounds(u.c) )


    def test_upsamplingvec_subvector(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                offset = int(m / 4)
                msub = int(m / 2)
                ksub = int(k / 2)

                usub = lib.upsamplingvec()
                usub_py, usub_ptr = gen_py_upsamplingvec(lib, msub, k)
                with UpsamplingVecContext(lib, m, k, random=True) as u:
                    assert NO_ERR( lib.upsamplingvec_subvector(
                            usub, u.c, offset, msub, k) )
                    assert NO_ERR( lib.indvector_memcpy_av(
                            usub_ptr, usub.vec, 1) )
                    assert ( usub.size1 == msub )
                    assert ( usub.size2 == k )
                    assert VEC_EQ(usub_py, u.py[offset : offset+msub], 0, 0)

                    assert NO_ERR( lib.upsamplingvec_subvector(
                            usub, u.c, offset, msub, ksub) )
                    assert ( usub.size1 == msub )
                    assert ( usub.size2 == ksub )


    def test_upsamplingvec_mul_matrix(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n, True)
                ATOLK = RTOL * k**0.5

                rowmajor = lib.enums.CblasRowMajor
                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, rowmajor, random=True)
                B = okcctx.CDenseMatrixContext(lib, n, m, rowmajor, random=True)
                C = okcctx.CDenseMatrixContext(lib, k, n, rowmajor, random=True)
                D = okcctx.CDenseMatrixContext(lib, n, k, rowmajor, random=True)
                E = lib.matrix(0, 0, 0, None, 0)

                mvec = okcctx.CVectorContext(lib, m)
                nvec = okcctx.CVectorContext(lib, n)
                kvec = okcctx.CVectorContext(lib, k)
                u = UpsamplingVecContext(lib, m, k, random=True)

                T = lib.enums.CblasTrans
                N = lib.enums.CblasNoTrans
                alpha = np.random.rand()
                beta = np.random.rand()

                with A, B, C, D, mvec, nvec, kvec, u, hdl as hdl:
                    # functioning cases
                    upsamplingvec_mul(
                            'N', 'N', 'N', alpha, u.py, C.py, beta, A.py)
                    assert NO_ERR(lib.upsamplingvec_mul_matrix(
                            hdl, N, N, N, alpha, u.c, C.c, beta, A.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, A.c, nvec.c, 0, mvec.c) )
                    mvec.sync_to_py()
                    assert VEC_EQ( mvec.py, A.py.dot(nvec.py), ATOLM, RTOL )

                    upsamplingvec_mul(
                            'T', 'N', 'N', alpha, u.py, A.py, beta, C.py)
                    assert NO_ERR( lib.upsamplingvec_mul_matrix(
                            hdl, T, N, N, alpha, u.c, A.c, beta, C.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, C.c, nvec.c, 0, kvec.c) )
                    kvec.sync_to_py()
                    assert VEC_EQ( kvec.py, C.py.dot(nvec.py), ATOLK, RTOL )

                    upsamplingvec_mul(
                            'N', 'N', 'T', alpha, u.py, C.py, beta, B.py)
                    assert NO_ERR( lib.upsamplingvec_mul_matrix(
                            hdl, N, N, T, alpha, u.c, C.c, beta, B.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, B.c, mvec.c, 0, nvec.c) )
                    nvec.sync_to_py()
                    assert VEC_EQ( nvec.py, B.py.dot(mvec.py), ATOLN, RTOL )

                    upsamplingvec_mul(
                            'T', 'N', 'T', alpha, u.py, A.py, beta, D.py)
                    assert NO_ERR( lib.upsamplingvec_mul_matrix(
                            hdl, T, N, T, alpha, u.c, A.c, beta, D.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, D.c, kvec.c, 0, nvec.c) )
                    nvec.sync_to_py()
                    assert VEC_EQ( nvec.py, D.py.dot(kvec.py), ATOLN, RTOL )

                    upsamplingvec_mul(
                            'N', 'T', 'N', alpha, u.py, D.py, beta, A.py)
                    assert NO_ERR( lib.upsamplingvec_mul_matrix(
                            hdl, N, T, N, alpha, u.c, D.c, beta, A.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, A.c, nvec.c, 0, mvec.c) )
                    mvec.sync_to_py()
                    assert VEC_EQ( mvec.py, A.py.dot(nvec.py), ATOLM, RTOL )

                    upsamplingvec_mul(
                            'T', 'T', 'N', alpha, u.py, B.py, beta, C.py)
                    assert NO_ERR( lib.upsamplingvec_mul_matrix(
                            hdl, T, T, N, alpha, u.c, B.c, beta, C.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, C.c, nvec.c, 0, kvec.c) )
                    kvec.sync_to_py()
                    assert VEC_EQ( kvec.py, C.py.dot(nvec.py), ATOLK, RTOL)

                    upsamplingvec_mul(
                            'N', 'T', 'T', alpha, u.py, D.py, beta, B.py)
                    assert NO_ERR( lib.upsamplingvec_mul_matrix(
                            hdl, N, T, T, alpha, u.c, D.c, beta, B.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, B.c, mvec.c, 0, nvec.c) )
                    nvec.sync_to_py()
                    assert VEC_EQ( nvec.py, B.py.dot(mvec.py), ATOLN, RTOL)

                    upsamplingvec_mul(
                            'T', 'T', 'T', alpha, u.py, B.py, beta, D.py)
                    assert NO_ERR( lib.upsamplingvec_mul_matrix(
                            hdl, T, T, T, alpha, u.c, B.c, beta, D.c) )
                    assert NO_ERR( lib.blas_gemv(hdl, N, 1, D.c, kvec.c, 0, nvec.c) )
                    nvec.sync_to_py()
                    assert VEC_EQ( nvec.py, D.py.dot(kvec.py), ATOLN, RTOL)

                    # reject: dimension mismatch
                    print('\nexpect dimension mismatch error:')
                    err = lib.upsamplingvec_mul_matrix(
                            hdl, N, N, N, alpha, u.c, C.c, beta, B.c)
                    assert ( err == lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )
                    print('\nexpect dimension mismatch error:')
                    err = lib.upsamplingvec_mul_matrix(
                            hdl, T, N, N, alpha, u.c, A.c, beta, D.c)
                    assert ( err == lib.enums.OPTKIT_ERROR_DIMENSION_MISMATCH )

                    # reject: unallocated
                    print('\nexpect unallocated error:')
                    err = lib.upsamplingvec_mul_matrix(
                            hdl, T, N, N, alpha, u.c, E, beta, C.c)
                    assert ( err == lib.enums.OPTKIT_ERROR_UNALLOCATED )

    def test_upsamplingvec_count(self):
        m, n = defs.shape()
        k = self.k
        TOL = 1e-7
        ATOL = TOL * k**0.5
        for lib in self.libs:
            with lib as lib:
                hdl = okcctx.CDenseLinalgContext(lib)
                u = UpsamplingVecContext(lib, m, k, random=True)
                counts = okcctx.CVectorContext(lib, k)
                with u, counts, hdl as hdl:
                    assert NO_ERR( lib.upsamplingvec_count(u.c, counts.c) )
                    counts.sync_to_py()
                    counts_host = np.zeros(k)
                    for idx in u.py:
                        counts_host[idx] += 1
                    assert VEC_EQ(counts.py, counts_host, ATOL, TOL )

    def test_cluster_aid_alloc_free(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                h = lib.cluster_aid()
                try:
                    assert NO_ERR( lib.cluster_aid_alloc(h, m, k, lib.enums.CblasRowMajor) )
                    assert ( h.a2c_tentative_full.size1 == m )
                    assert ( h.a2c_tentative_full.size2 == k )
                    assert ( h.a2c_tentative.size1 == m )
                    assert ( h.a2c_tentative.size2 == k )
                    assert ( h.d_min_full.size == m )
                    assert ( h.d_min.size == m )
                    assert ( h.c_squared_full.size == k )
                    assert ( h.c_squared.size == k )
                    assert ( h.D_full.size1 == k )
                    assert ( h.D_full.size2 == m )
                    assert ( h.D.size1 == k )
                    assert ( h.D.size2 == m )
                    assert ( h.reassigned == 0 )
                finally:
                    assert NO_ERR( lib.cluster_aid_free(h) )

                assert ( h.a2c_tentative_full.size1 == 0 )
                assert ( h.a2c_tentative_full.size2 == 0 )
                assert ( h.d_min_full.size == 0 )
                assert ( h.c_squared_full.size == 0 )
                assert ( h.D_full.size1 == 0 )
                assert ( h.D_full.size2 == 0 )

    def test_cluster_aid_resize(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                pass
                # TODO: FINISH THIS TEST?

    def test_kmeans_work_alloc_free(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                w = lib.kmeans_work()
                try:
                    assert NO_ERR( lib.kmeans_work_alloc(w, m, k, n) )
                    assert ( w.n_vectors == m )
                    assert ( w.n_clusters == k )
                    assert ( w.vec_length == n )
                    assert ( w.A.size1 == m )
                    assert ( w.A.size2 == n )
                    assert ( w.A.data != 0 )
                    assert ( w.C.size1 == k )
                    assert ( w.C.size2 == n )
                    assert ( w.C.data != 0 )
                    assert ( w.counts.size == k )
                    assert ( w.counts.data != 0 )
                    assert ( w.a2c.size1 == m )
                    assert ( w.a2c.size2 == k )
                    assert ( w.a2c.indices != 0 )
                finally:
                    assert NO_ERR( lib.kmeans_work_free(w) )

                assert ( w.n_vectors == 0 )
                assert ( w.n_clusters == 0 )
                assert ( w.vec_length == 0 )

    def test_kmeans_work_resize(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                with KmeansWorkContext(lib, m, k, n) as w:
                    assert ( w.n_vectors == m )
                    assert ( w.n_clusters == k )
                    assert ( w.vec_length == n )
                    assert ( w.A.size1 == m )
                    assert ( w.A.size2 == n )
                    assert ( w.C.size1 == k )
                    assert ( w.C.size2 == n )
                    assert ( w.counts.size == k )
                    assert ( w.a2c.size1 == m )
                    assert ( w.a2c.size2 == k )

                    assert NO_ERR( lib.kmeans_work_subselect(
                            w, int(m/2), int(k/2), int(n/2)) )

                    assert ( w.n_vectors == m )
                    assert ( w.n_clusters == k )
                    assert ( w.vec_length == n )
                    assert ( w.A.size1 == int(m / 2) )
                    assert ( w.A.size2 == int(n / 2) )
                    assert ( w.C.size1 == int(k / 2) )
                    assert ( w.C.size2 == int(n / 2) )
                    assert ( w.counts.size == int(k / 2) )
                    assert ( w.a2c.size1 == int(m / 2) )
                    assert ( w.a2c.size2 == int(k / 2) )


    def test_kmeans_work_load_extract(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, _, ATOLMN = STANDARD_TOLS(lib, m, n, True)
                ATOLKN = RTOL * (k*n)**0.5
                ATOLK = RTOL * k**0.5
                orderA = orderC = lib.enums.CblasRowMajor

                A, A_ptr = okcctx.gen_py_matrix(lib, m, n, orderA, random=True)
                C, C_ptr = okcctx.gen_py_matrix(lib, k, n, orderC, random=True)
                a2c, a2c_ptr = gen_py_upsamplingvec(lib, m, k, random=True)
                counts, counts_ptr = okcctx.gen_py_vector(lib, k)

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

                with KmeansWorkContext(lib, m, k, n) as w:
                    assert NO_ERR( lib.kmeans_work_load(
                            w, A_ptr, orderA, C_ptr, orderC, a2c_ptr, 1,
                            counts_ptr, 1) )

                    assert NO_ERR(lib.matrix_scale(w.C, SCALING))
                    assert NO_ERR(lib.vector_scale(w.counts, SCALING))

                    assert NO_ERR( lib.kmeans_work_extract(
                            C_ptr, orderC, a2c_ptr, 1, counts_ptr, 1, w) )

                    assert VEC_EQ( A_orig, A, ATOLKN, RTOL )
                    assert VEC_EQ( C_orig, C, ATOLKN, RTOL )
                    assert VEC_EQ( a2c_orig, a2c, ATOLM, RTOL )
                    assert VEC_EQ( counts_orig, counts, ATOLK, RTOL )

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
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n, True)
                ATOLK = RTOL * k**0.5
                rowmajor = lib.enums.CblasRowMajor

                for MAXDIST in [1e3, 0.2]:
                    hdl = okcctx.CDenseLinalgContext(lib)
                    A = okcctx.CDenseMatrixContext(lib, m, n, rowmajor, value=self.A_test)
                    C = okcctx.CDenseMatrixContext(lib, k, n, rowmajor, value=self.C_test)
                    a2c = UpsamplingVecContext(lib, m, k, random=True)
                    h = ClusterAidContext(lib, m, k, rowmajor)

                    mvec = okcctx.CVectorContext(lib, m, random=True)
                    kvec = okcctx.CVectorContext(lib, k)

                    a2c_orig = np.zeros(m).astype(ct.c_size_t)
                    a2c_orig[:] = a2c.py[:]

                    with A, C, a2c, mvec, kvec, h as h, hdl as hdl:
                        assert NO_ERR( lib.cluster(A.c, C.c, a2c.c, h, MAXDIST) )

                        D, dmin, reassigned = cluster(A.py, C.py, a2c.py, MAXDIST)

                        # verify number reassigned
                        a2c.sync_to_py()
                        assert ( h.reassigned == sum(a2c.py != a2c_orig) )
                        assert ( h.reassigned == reassigned )

                        # verify all distances
                        assert NO_ERR( lib.blas_gemv(
                                hdl, lib.enums.CblasNoTrans, 1, h.D_full,
                                mvec.c, 0, kvec.c) )
                        kvec.sync_to_py()
                        assert VEC_EQ( D.dot(mvec.py), kvec.py, ATOLK, RTOL)

                        # verify min distances
                        dmin_py = np.zeros(m).astype(lib.pyfloat)
                        dmin_ptr = dmin_py.ctypes.data_as(lib.ok_float_p)
                        assert NO_ERR( lib.vector_memcpy_av(
                                    dmin_ptr, h.d_min_full, 1) )
                        assert VEC_EQ( dmin_py, dmin, ATOLM, RTOL )

    def test_calculate_centroids(self):
        """ calculate centroids

            given matrix A, upsampling vector a2c,
            calculate centroid matrix C

            compare C * xrand in Python vs. C
        """
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n, True)
                ATOLK = RTOL * k**0.5
                orderA = orderC = lib.enums.CblasRowMajor

                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, orderA, value=self.A_test)
                C = okcctx.CDenseMatrixContext(lib, k, n, orderC, random=True)
                a2c = UpsamplingVecContext(lib, m, k, random=True)
                counts = okcctx.CVectorContext(lib, k)
                nvec = okcctx.CVectorContext(lib, n, random=True)
                kvec = okcctx.CVectorContext(lib, k)
                h = ClusterAidContext(lib, m, k, orderA)

                with A, C, a2c, counts, nvec, kvec, hdl as hdl, h as h:
                    # C: build centroids
                    assert NO_ERR( lib.calculate_centroids(
                            A.c, C.c, a2c.c, counts.c, h) )

                    # Py: build centroids
                    upsamplingvec_mul('T', 'N', 'N', 1, a2c.py, A.py, 0, C.py)
                    counts_local = np.zeros(k)
                    for c in a2c.py:
                        counts_local[c] += 1
                    for idx, c in enumerate(counts_local):
                        counts_local[idx] = (1. / c) if c > 0 else 0.
                        C.py[idx, :] *= counts_local[idx]

                    # kvec =?= C * nvec
                    assert NO_ERR( lib.blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, C.c, nvec.c, 0,
                            kvec.c) )
                    kvec.sync_to_py()
                    assert VEC_EQ( kvec.py, C.py.dot(nvec.py), ATOLK, RTOL)


    def test_kmeans(self):
        """ k-means

            given matrix A, cluster # k

            cluster A by kmeans algorithm. compare Python vs. C
        """
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                orderA = orderC = lib.enums.CblasRowMajor
                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, orderA, value=self.A_test)
                C = okcctx.CDenseMatrixContext(lib, k, n, orderC, random=True)
                a2c = UpsamplingVecContext(lib, m, k, random=True)
                counts = okcctx.CVectorContext(lib, k)
                h = ClusterAidContext(lib, m, k, orderA)

                DIST_RELTOL = 0.1
                CHANGE_TOL = int(1 + 0.01 * m)
                MAXITER = 500
                VERBOSE = 1

                with A, C, a2c, counts, hdl as hdl, h as h:
                    assert NO_ERR( lib.k_means(
                            A.c, C.c, a2c.c, counts.c, h, DIST_RELTOL,
                            CHANGE_TOL, MAXITER, VERBOSE) )

    def test_kmeans_easy_init_free(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                work = None
                w_int = 0
                try:
                    w_int = lib.kmeans_easy_init(m, k, n)
                    self.assertNotEqual(w_int, 0)
                    work = ct.c_void_p(w_int)
                    cwork = ct.cast(work, lib.kmeans_work_p)
                    assert ( cwork.contents.n_vectors == m )
                    assert ( cwork.contents.n_clusters == k )
                    assert ( cwork.contents.vec_length == n )
                finally:
                    if work is None:
                        work = ct.c_void_p(w_int)
                    assert NO_ERR( lib.kmeans_easy_finish(work) )

    def test_kmeans_easy_resize(self):
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                with okcctx.CPointerContext(
                        lambda: lib.kmeans_easy_init(m, k, n),
                        lib.kmeans_easy_finish) as work:
                    assert NO_ERR( lib.kmeans_easy_resize(
                            work, int(m/2), int(k/2), int(n/2)) )
                    cwork = ct.cast(work, lib.kmeans_work_p)
                    assert ( cwork.contents.A.size1 == int(m / 2) )
                    assert ( cwork.contents.C.size1 == int(k / 2) )
                    assert ( cwork.contents.A.size2 == int(n / 2) )

    def test_kmeans_easy_run(self):
        """ k-means (unified call)

            given matrix A, cluster # k

            cluster A by kmeans algorithm. compare Python vs. C
        """
        m, n = defs.shape()
        k = self.k
        for lib in self.libs:
            with lib as lib:
                orderA = orderC = lib.enums.CblasRowMajor
                A, A_ptr = okcctx.gen_py_matrix(lib, m, n, orderA)
                A += self.A_test

                C, C_ptr = okcctx.gen_py_matrix(lib, k, n, orderC, random=True)
                a2c, a2c_ptr = gen_py_upsamplingvec(lib, m, k, random=True)
                counts, counts_ptr = okcctx.gen_py_vector(lib, k)

                with okcctx.CPointerContext(
                        lambda: lib.kmeans_easy_init(m, k, n),
                        lib.kmeans_easy_finish) as work:

                    io = lib.kmeans_io(
                            A_ptr, C_ptr, counts_ptr, a2c_ptr, orderA, orderC,
                            1, 1)

                    DIST_RELTOL = 0.1
                    CHANGE_TOL = int(1 + 0.01 * m)
                    MAXITER = 500
                    VERBOSE = 1
                    settings = lib.kmeans_settings(
                            DIST_RELTOL, CHANGE_TOL, MAXITER, VERBOSE)

                    assert NO_ERR( lib.kmeans_easy_run(work, settings, io) )
