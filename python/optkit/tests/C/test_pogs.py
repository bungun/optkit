from optkit.compat import *

import os
import itertools
import unittest

import optkit.libs.enums as enums
from optkit.libs.pogs import PogsDenseLibs
from optkit.tests.defs import OptkitTestCase
from optkit.tests import defs
import optkit.tests.C.statements as okctest
import optkit.tests.C.context_managers as okcctx
from optkit.tests.C.pogs_contexts import Dense as DenseTest
import optkit.tests.C,pogs_statements as pogs_test


class PogsDenseLibsTestCase(unittest.TestCase):
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
        assert any([self.libs.get(*c) for c in defs.LIB_CONDITIONS])

class PogsDenseTestCase(unittest.TestCase):
    """TODO: docstring"""
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        libs = map(lambda c: okcctx.CLibContext(
                None, PogsDenseLibs(), c), defs.LIB_CONDITIONS)
        self.libs = list(filter(lambda ctx: ctx.lib is not None, libs))
        self.A_test = defs.A_test_gen()

    def setUp(self):
        objectives = ['Abs', 'AbsQuad', 'AbsExp', 'AsymmSquare']
        self.LIBS_LAYOUTS = list(itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS))
        self.LIBS_LAYOUTS_OBJECTIVES = list(itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS, objectives))

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_pogs_dense_default_settings(self):
        for lib in self.libs:
            with lib as lib:
                assert pogs_test.settings_are_default(lib)

    def test_pogs_dense_init_finish(self, reset=0):
        m, n = defs.shape()
        for (lib, layout) in self.LIBS_LAYOUTS:
            with lib as lib:
                A_py, A_ptr  = okcctx.gen_py_matrix(lib, m, n, layout)
                A_py += self.A_test
                flags = lib.pogs_solver_flags(m, n, layout)
                solver = lib.pogs_init(A_ptr, flags)
                assert okctest.noerr(lib.pogs_finish(solver, 0))

    def test_pogs_dense_components(self):
        for (lib, layout, obj) in self.LIBS_LAYOUTS_OBJECTIVES:
            with DenseTest(lib, self.A_test, layout, obj) as ctx:
                assert pogs_test.solver_components_work(ctx)

    def test_pogs_dense_call(self):
        for (lib, layout, obj) in self.LIBS_LAYOUTS_OBJECTIVES:
            with DenseTest(lib, self.A_test, layout, obj) as ctx:
                assert pogs_test.solve_call_executes(ctx)

    def test_pogs_dense_accelerate(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with DenseTest(lib, self.A_test, layout) as ctx:
                assert pogs_test.anderson_reduces_iterations(ctx)

    def test_pogs_dense_call_unified(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with DenseTest(lib, self.A_test, layout, solver=False) as ctx:
                assert pogs_test.integrated_pogs_call_executes(ctx)

    def test_pogs_dense_warmstart_scaling(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with DenseTest(lib, self.A_test, layout) as ctx:
                assert pogs_test.solver_scales_warmstart_inputs(ctx)

    def test_pogs_dense_warmstart(self):
        for (lib, layout) in self.LIBS_LAYOUTS:
            with DenseTest(lib, self.A_test, layout) as ctx:
                assert pogs_test.warmstart_reduces_iterations(ctx)

#     def test_pogs_dense_io(self):
#         for (lib, layout) in self.LIBS_LAYOUTS:
#             with DenseTest(lib, self.A_test, layout) as ctx:
#                 assert pogs_test.solver_data_transferable(ctx)

