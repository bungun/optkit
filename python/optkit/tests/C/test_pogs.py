from optkit.compat import *

import os
import numpy as np
import ctypes as ct
import itertools

import optkit.libs.enums as enums
from optkit.libs.pogs import PogsDenseLibs
from optkit.tests.defs import OptkitTestCase
import optkit.tests.C.statements as okctest
import optkit.tests.C.context_managers as okcctx
from optkit.tests.C.pogs_base import PogsBaseTestContext, OptkitCPogsTestCase

MATRIX_ORDERS = enums.OKEnums.MATRIX_ORDERS

class PogsDenseTestContext(PogsBaseTestContext):
    def __init__(self, libctx, A, layout, obj='Abs', solver=True):
        m, n = A.shape
        lib = libctx.lib
        PogsBaseTestContext.__init__(self, libctx, m, n, obj=obj)
        order = 'C' if layout == lib.enums.CblasRowMajor else 'F'
        self.A_dense = np.zeros((m, n), dtype=lib.pyfloat, order=order)
        self.A_dense += A
        self.data = self.A_dense.ctypes.data_as(lib.ok_float_p)
        self.flags = lib.pogs_solver_flags(m, n, layout)
        self._solver = solver

    def __enter__(self):
        PogsBaseTestContext.__enter__(self)
        if self._solver:
            self.solver = self.lib.pogs_init(self.data, self.flags)
        return self

    def __exit__(self, *exc):
        if self._solver:
            assert okctest.noerr(self.lib.pogs_finish(self.solver, 0))
        PogsBaseTestContext.__exit__(self, *exc)

class PogsDenseLibsTestCase(OptkitTestCase):
    """TODO: docstring"""
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = PogsDenseLibs()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_libs_exist(self):
        self.assertTrue( any([self.libs.get(*c) for c in self.CONDITIONS]) )

class PogsDenseTestCase(OptkitCPogsTestCase):
    """TODO: docstring"""
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        libs = map(lambda c: okcctx.CLibContext(
                None, PogsDenseLibs(), c), self.CONDITIONS)
        self.libs = filter(lambda ctx: ctx.lib is not None, libs)
        self.A_test = self.A_test_gen

    def tearDown(self):
        self.free_all_vars()
        self.exit_call()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_pogs_dense_default_settings(self):
        for lib in self.libs:
            with lib as lib:
                self.assert_default_settings(lib)

    def test_pogs_dense_init_finish(self, reset=0):
        m, n = self.shape
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with lib as lib:
                A_py, A_ptr  = okcctx.gen_py_matrix(lib, m, n, layout)
                A_py += self.A_test
                flags = lib.pogs_solver_flags(m, n, layout)
                solver = lib.pogs_init(A_ptr, flags)
                okctest.assert_noerr(lib.pogs_finish(solver, 0))

    def test_pogs_dense_components(self):
        objs = ['Abs', 'AbsQuad', 'AbsExp', 'AsymmSquare']
        for (lib, layout, obj) in itertools.product(
                self.libs, MATRIX_ORDERS, objs):
            with PogsDenseTestContext(lib, self.A_test, layout, obj) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_coldstart_components(ctx)

    def test_pogs_dense_call(self):
        objs = ['Abs', 'AbsQuad', 'AbsExp', 'AsymmSquare']
        for (lib, layout, obj) in itertools.product(
                self.libs, MATRIX_ORDERS, objs):
            with PogsDenseTestContext(lib, self.A_test, layout, obj) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_call(ctx)

    def test_pogs_dense_accelerate(self):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with PogsDenseTestContext(lib, self.A_test, layout) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_accelerate(ctx)

    def test_pogs_dense_call_unified(self):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with PogsDenseTestContext(
                    lib, self.A_test, layout, solver=False) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_unified(ctx)

    def test_pogs_dense_warmstart_scaling(self):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with PogsDenseTestContext(lib, self.A_test, layout) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_warmstart_scaling(ctx)

    def test_pogs_dense_warmstart(self):
        for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
            with PogsDenseTestContext(lib, self.A_test, layout) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_warmstart(ctx)

#     def test_pogs_dense_io(self):
#         for (lib, layout) in itertools.product(self.libs, MATRIX_ORDERS):
#             with PogsDenseTestContext(lib, self.A_test, layout) as ctx:
#                 self.assert_pogs_io(ctx)

