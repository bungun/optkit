from optkit.compat import * # import contextlib

import os
import numpy as np
import ctypes as ct
import itertools
import collections

import optkit.libs.enums as enums
from optkit.libs.pogs import PogsLibs
import optkit.tests.C.base_layer as oktest
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.pogs_base import PogsBaseTestContext, OptkitCPogsTestCase

MATRIX_ORDERS = enums.OkEnums.MATRIX_ORDERS

class PogsDenseTestContext(PogsBaseTestContext):
    def __init__(self, lib, A, layout, obj='Abs', solver=True):
        m, n = A.shape
        PogsBaseTestContext.__init__(self, lib, m, n, obj=obj)
        self.A_dense = A
        order = 'C' if layout == lib.enums.CblasRowMajor else 'F'
        self.data = np.ravel(A, order=order).ctypes.data_as(lib.ok_float_p)
        self.flags = lib.pogs_solver_flags(m, n, layout)
        self._solver = solver

    def __enter__(self):
        PogsBaseTestContext.__enter__(self)
        if self._solver:
            self.solver = self.lib.pogs_init(self.data, self.flags)
        return self

    def __exit__(self):
        if self._solver:
            oktest.assert_noerr(self.lib.pogs_free(self.solver))
        PogsBaseTestContext.__exit__(self)

class PogsDenseLibsTestCase(OptkitTestCase):
    """TODO: docstring"""
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = PogsLibs()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_libs_exist(self):
        self.assertTrue( any([self.libs.get(**c) for c in self.CONDITIONS]) )

class PogsDenseTestCase(OptkitCPogsTestCase):
    """TODO: docstring"""
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        libs = map(lambda c: oktest.CLibContext(
                None, PogsLibs(), c), self.CONDITIONS)
        self.libs = filter(lambda ctx: ctx.lib is not None, libs)
        self.A_test = self.A_test_gen
        m, n = self.m, self.n = self.shape
        self.RTOL = 10**(7 - 2 * lib.FLOAT)
        self.ATOLM = RTOL * m**0.5
        self.ATOLN = RTOL * n**0.5
        self.ATOLMN = RTOL * (m + n)**0.5

    def tearDown(self):
        self.free_all_vars()
        self.exit_call()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_pogs_dense_default_settings(self):
        for lib in self.libs:
            with CLibContext(lib) as lib:
                self.assert_default_settings(lib)

    def test_pogs_dense_init_finish(self, reset=0):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with contextlib.ExitStack() as cm:
                lib = cm.enter_context(lib)
                A = cm.enter_context(oktest.CDenseMatrixContext(lib, layout))
                A.copy_to_c(self.A_test)
                flags = lib.pogs_solver_flags(m, n, order)
                solver = lib.pogs_init(A.ptr, A.flags)
                oktest.assert_noerr(lib.pogs_finish(solver, 0))

    def test_pogs_dense_components(self):
        objs = ['Abs', 'AbsQuad', 'AbsExp', 'AsymmSquare']
        for (lib, layout, obj) in itertools.product(
                self.libs, MATRIX_ORDERS, objs):
            with PogsDenseTestContext(lib, self.A_test, layout, obj) as ctx:
                self.assert_coldstart_components(ctx)

    def test_pogs_dense_call(self):
        objs = ['Abs', 'AbsQuad', 'AbsExp', 'AsymmSquare']
        for (lib, layout, obj) in itertools.product(
                self.libs, MATRIX_ORDERS, objs):
            with PogsDenseTestContext(lib, self.A_test, layout, obj) as test:
                self.assert_pogs_call(ctx)

    def test_pogs_dense_call_unified(self):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with PogsDenseTestContext(
                    lib, self.A_test, layout, solver=False) as test:
                self.assert_pogs_unified(ctx)

    def test_pogs_dense_warmstart(self):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with PogsDenseTestContext(lib, self.A_test, layout) as test:
                self.assert_pogs_warmstart(ctx)

    def test_pogs_dense_io(self):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with PogsDenseTestContext(lib, self.A_test, layout) as test:
                self.assert_pogs_io(ctx)

