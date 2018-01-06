from optkit.compat import *

import os
import numpy as np
import itertools
import unittest

import optkit.libs.enums as enums
from optkit.libs.pogs import PogsAbstractLibs
from optkit.tests import defs
from optkit.tests.C import statements
from optkit.tests.C import context_managers as okcctx
from optkit.tests.C.pogs_contexts import Abstract as AbstractTest
import optkit.tests.C.pogs_statements as pogs_test

NO_ERR = statements.noerr


class PogsAbstractTestCase(unittest.TestCase):
    """TODO: docstring"""
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(PogsAbstractLibs())
        self.A_test = defs.A_test_gen()
        m, n = self.m, self.n = defs.shape()

    def setUp(self):
        operators = ['dense', 'sparse']
        direct = (0 , 1)
        equil = (1., 2.)
        self.LIBS_OPS = list(itertools.product(self.libs, operators))
        self.LIBS_OPS_DIRECT = list(itertools.product(
                self.libs, operators, direct))
        self.LIBS_OPS_DIRECT_EQUIL = list(itertools.product(
                self.libs, operators, direct, equil))

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_libs_exist(self):
        assert any(self.libs)

    def test_pogs_abstract_operator_gen_free(self):
        for (lib, op) in self.LIBS_OPS:
            with lib as lib:
                with okcctx.c_operator_context(lib, op, self.A_test) as o:
                    assert isinstance(o, lib.abstract_operator_p)

    def test_pogs_abstract_default_settings(self):
        for lib in self.libs:
            with lib as lib:
                assert pogs_test.settings_are_default(lib)

    def test_pogs_abstract_init_finish(self, reset=0):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            with lib as lib:
                with okcctx.c_operator_context(lib, op, self.A_test) as o:
                    flags = lib.pogs_solver_flags(direct, equil)
                    solver = lib.pogs_init(o, flags)
                    assert NO_ERR(lib.pogs_finish(solver, 0))

    def test_pogs_abstract_components(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            if op == 'sparse' and direct:
                continue
            with AbstractTest(lib, self.A_test, op, direct, equil) as ctx:
                assert pogs_test.solver_components_work(ctx)

    def test_pogs_abstract_call(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            if op == 'sparse' and direct:
                continue
            with AbstractTest(lib, self.A_test, op, direct, equil) as ctx:
                assert pogs_test.solve_call_executes(ctx)

    def test_pogs_abstract_diagnostic(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            if op == 'sparse' and direct:
                continue
            with AbstractTest(lib, self.A_test, op, direct, equil) as ctx:
                assert pogs_test.residuals_recoverable(ctx)

    def test_pogs_abstract_accelerate(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            if op == 'sparse' and direct:
                continue
            defs.verbose_print('\nOPERATOR={}'.format(op))
            defs.verbose_print('DIRECT={}'.format(direct))
            defs.verbose_print('EQUILNORM={}'.format(equil))
            with AbstractTest(
                    lib, self.A_test, op, direct, equil, obj='Logistic') as ctx:

                ctx.params.settings.maxiter = 1000

                # OVERRELAXATION
                assert pogs_test.overrelaxation_reduces_iterations(ctx)

                # ADAPTIVE RHO
                if direct:
                    assert pogs_test.adaptive_rho_reduces_iterations(ctx)

                # ANDERSON ACCELERATION
                if equil == 2:
                    assert pogs_test.anderson_reduces_iterations(ctx)

                # EXTRATOL
                if equil == 1 and not direct:
                    assert pogs_test.extratol_reduces_iterations(ctx)

    def test_pogs_abstract_call_unified(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            if op == 'sparse' and direct:
                continue
            with AbstractTest(lib, self.A_test, op, direct, equil) as ctx:
                assert pogs_test.integrated_pogs_call_executes(ctx)

    def test_pogs_abstract_warmstart(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            if op == 'sparse' and direct:
                continue
            defs.verbose_print('\nOPERATOR={}'.format(op))
            defs.verbose_print('DIRECT={}'.format(direct))
            defs.verbose_print('EQUILNORM={}'.format(equil))
            with AbstractTest(lib, self.A_test, op, direct, equil) as ctx:
                assert pogs_test.solver_scales_warmstart_inputs(ctx)
                # assert pogs_test.warmstart_reduces_iterations(ctx)

    def test_pogs_abstract_io(self):
        pass
        # # TODO: FIX
        # equil = 1.
        # direct = 0
        # for (lib, op) in self.LIBS_OPS:
        #     with AbstractTest(lib, self.A_test, op, direct, equil) as ctx:
        #         assert pogs_test.solver_data_transferable(ctx)