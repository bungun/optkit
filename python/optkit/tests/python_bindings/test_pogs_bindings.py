from optkit.compat import *

import gc
import os
import numpy as np
import scipy.sparse as sp
import unittest

from optkit import api
from optkit.types import pogs
from optkit.tests import defs
from optkit.tests import statements
from optkit.tests import context_managers as okctx

SCAL_EQ = statements.scalar_equal

if os.getenv('OPTKIT_PYTEST_GPU', False):
    optkit.set_backend(gpu=1)

print('BACKEND:', api.backend.config)

class PogsBindingsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.__files = []

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig
        for f in self.__files:
            if os.path.exists(f):
                os.remove(f)

    def setUp(self):
        self.shape = defs.shape()
        self.A_test = defs.A_test_gen()
        self.A_test_csr = sp.csr_matrix(defs.A_test_sparse_gen())
        self.A_test_csc = sp.csc_matrix(defs.A_test_sparse_gen())
        self.test_matrices = [self.A_test, self.A_test_csc, self.A_test_csr]

    def register_file(self, file):
        if os.path.exists(file):
            self.__files.append(file)

    def test_objective(self):
        m = self.shape[0]
        f = api.PogsObjective(m, h='Abs', b=1, d=2, e=3)
        h, a, b, c, d, e, s = f.arrays

        for i in xrange(m):
            assert (h[i] == f.enums.dict['Abs'])
            assert (a[i] == 1.)
            assert (b[i] == 1.)
            assert (c[i] == 1.)
            assert (d[i] == 2.)
            assert (e[i] == 3.)
            assert (s[i] == 1.)

        # set block by scalar
        f.set(end=int(m / 2), b=0.5)
        for i in xrange(int(m / 2)):
            assert (f.b[i] == 0.5)

        # set block by vector
        c = np.random.rand(int(m / 4))
        f.set(start=int(m / 2), end=int(m / 2) + int(m / 4), c=c)
        for i, idx in enumerate(
                xrange(int(m / 2), int(m / 2) + int(m / 4))):
            assert (f.c[idx] == c[i])

        # set indices by scalar
        indices = []
        threshold = lambda i: i if np.random.rand() > 0.75 else False
        while len(indices) == 0:
            indices = listfilter(None, listmap(threshold, xrange(m)))
        f.set(range=indices, e=0.1)
        for idx in indices:
            assert (f.e[idx] == 0.1)

        # set indices by vector
        d = np.random.rand(len(indices))
        f.set(range=indices, d=d)
        for i, idx in enumerate(indices):
            assert (f.d[idx] == d[i])

        # block copy
        msub = int(m / 3)
        fsub = api.PogsObjective(
                msub, b=np.random.rand(msub), d=np.random.rand(msub))
        f.copy_from(fsub)
        assert ( sum(f.b[:msub] - fsub.b) == 0 )
        assert ( sum(f.d[:msub] - fsub.d) == 0 )

        start_idx = int(m / 2)
        f.copy_from(fsub, start_idx)
        assert ( sum(f.b[start_idx:start_idx + msub] - fsub.b) == 0 )
        assert ( sum(f.d[start_idx:start_idx + msub] - fsub.d) == 0 )

    def test_solver_object(self):
        with api.PogsSolver(self.A_test):
            assert ( not api.backend.device_reset_allowed )
        gc.collect()
        assert ( api.backend.device_reset_allowed )

    def test_abstract_solver_object(self):
        for A in (self.A_test, self.A_test_csc, self.A_test_csr):
            with api.PogsAbstractSolver(A):
                assert( not api.backend.device_reset_allowed )
            gc.collect()
            assert ( api.backend.device_reset_allowed )

    def exercise_solve_call(self, solver_constructor, test_matrix):
        # objectives = ('Abs', 'AbsQuad', 'AbsExp', 'AsymmSquare')
        objectives = ['Abs']
        for h in objectives:
            if defs.VERBOSE_TEST:
                print('objective:', h)
            asymm = 2. if h != 'Abs' else 1.
            s = solver_constructor(test_matrix)
            f = api.PogsObjective(self.shape[0], h=h, b=1, s=asymm)
            g = api.PogsObjective(self.shape[1], h='IndGe0')
            s.solve(f, g)
            self.assertEqual(s.info.err, 0)
            self.assertTrue(
                    s.info.converged or s.info.iters == s.settings.maxiter)
            del s

    @staticmethod
    def transfer_success(solver_old, solver_new, TOL):
        obj_old = solver_old.info.c.obj
        obj_new = solver_new.info.c.obj
        assert (
            SCAL_EQ(obj_old, obj_new, TOL)
            or not solver_old.info.converged
            or not solver_new.info.converged)
        return True

    def test_solve_call(self):
        print(api.backend.config)
        self.exercise_solve_call(api.PogsSolver, self.A_test)

    def test_abstract_solve_call(self):
        N = len(self.test_matrices)
        for i, test_matrix in enumerate(self.test_matrices):
            t = type(test_matrix)
            print('\nabstract solve: {}/{}---{}'.format(i + 1, N, t))
            self.exercise_solve_call(api.PogsAbstractSolver, test_matrix)

    def test_solver_io(self):
        f = api.PogsObjective(self.shape[0], h='Abs', b=1)
        g = api.PogsObjective(self.shape[1], h='IndGe0')

        with api.PogsSolver(self.A_test) as s:
            factor = 10. * 2.**defs.TEST_ITERATE
            if api.backend.pogs.pyfloat == np.float32:
                factor *= 3
            tolerance = factor * s.settings.reltol
            def build_file():
                return s.save(os.path.abspath('.'), 'c_solve_test')
            with okctx.TempFileContext(build_file) as cache_file:
                s.solve(f, g, resume=0, maxiter=10000)
                with api.PogsSolver(self.A_test, no_init=1) as s2:
                    s2.load(cache_file)
                    s2.solve(f, g, resume=0, maxiter=10000)
                    assert self.transfer_success(s, s2, tolerance)
                    def build_file2():
                        return s2.save(os.path.abspath('.'), 'c_solve_test2')
                    with okctx.TempFileContext(build_file2) as cache_file2:
                        with api.PogsSolver(self.A_test, cache=cache_file2) as s3:
                            s3.solve(f, g, resume=1, maxiter=10000)
                            assert self.transfer_success(s, s3, tolerance)

