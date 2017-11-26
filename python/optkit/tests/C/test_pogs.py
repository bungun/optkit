from optkit.compat import *

import os
import itertools

import optkit.libs.enums as enums
from optkit.libs.pogs import PogsDenseLibs
from optkit.tests.defs import OptkitTestCase
import optkit.tests.C.statements as okctest
import optkit.tests.C.context_managers as okcctx
from optkit.tests.C import pogs_contexts
from optkit.tests.C.pogs_base import OptkitCPogsTestCase


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
        self.libs = list(filter(lambda ctx: ctx.lib is not None, libs))
        self.A_test = self.A_test_gen

    def setUp(self):
        objectives = ['Abs', 'AbsQuad', 'AbsExp', 'AsymmSquare']
        self.LIBS_LAYOUTS = list(itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS))
        self.LIBS_LAYOUTS_OBJECTIVES = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS, objectives)

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
        for (lib, layout) in self.LIBS_LAYOUTS:
            with lib as lib:
                A_py, A_ptr  = okcctx.gen_py_matrix(lib, m, n, layout)
                A_py += self.A_test
                flags = lib.pogs_solver_flags(m, n, layout)
                solver = lib.pogs_init(A_ptr, flags)
                assert okctest.noerr(lib.pogs_finish(solver, 0))

    def test_pogs_dense_components(self):
        for (lib, layout, obj) in self.LIBS_LAYOUTS_OBJECTIVES:
            with pogs_contexts.Dense(lib, self.A_test, layout, obj) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_coldstart_components(ctx)

    def test_pogs_dense_call(self):
        for (lib, layout, obj) in self.LIBS_LAYOUTS_OBJECTIVES:
            with pogs_contexts.Dense(lib, self.A_test, layout, obj) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_call(ctx)

    def test_pogs_dense_accelerate(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with pogs_contexts.Dense(lib, self.A_test, layout) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_accelerate(ctx)

    def test_pogs_dense_call_unified(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with pogs_contexts.Dense(
                    lib, self.A_test, layout, solver=False) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_unified(ctx)

    def test_pogs_dense_warmstart_scaling(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with pogs_contexts.Dense(lib, self.A_test, layout) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_warmstart_scaling(ctx)

    def test_pogs_dense_warmstart(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with pogs_contexts.Dense(lib, self.A_test, layout) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_warmstart(ctx)

#     def test_pogs_dense_io(self):
#         for (lib, layout) in self.LIBS_LAYOUTS:
#             with pogs_contexts.Dense(lib, self.A_test, layout) as ctx:
#                 self.assert_pogs_io(ctx)

