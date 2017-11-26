from optkit.compat import *

import os
import itertools

import optkit.libs.enums as enums
from optkit.libs.pogs import PogsLibs
from optkit.tests.defs import OptkitTestCase
import optkit.tests.C.statements as oktest
import optkit.test.C.context_managers as okcctx
from  optkit.tests.C import pogs_contexts
from optkit.tests.C.pogs_base import OptkitCPogsTestCase

class PogsAbstractLibsTestCase(OptkitTestCase):
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


class PogsAbstractTestCase(OptkitCPogsTestCase):
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

    def setUp(self):
        operators = ['dense', 'sparse']
        direct = (0 , 1)
        equil = (1., 2.)
        self.LIBS_OPS = list(itertools.product(self.libs, operators))
        self.LIBS_OPS_DIRECT = list(itertools.product(
                self.libs, operators, direct))
        self.LIBS_OPS_DIRECT_EQUIL = list(itertools.product(
                self.libs, operators, direct, equil))

    def tearDown(self):
        self.free_all_vars()
        self.exit_call()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_pogs_abstract_operator_gen_free(self):
        for (lib, op) in self.LIBS_OPS:
            with lib:
                with oktest.c_operator_context(lib, op, self.A_test) as op:
                    assert isinstance(o, lib.operator_p)

    def test_pogs_abstract_default_settings(self):
        for lib in self.libs:
            with CLibContext(lib) as lib:
                self.assert_default_settings(lib)

    def test_pogs_abstract_init_finish(self, reset=0):
        m, n = self.shape
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            with lib:
                with okcctx.c_operator_context(lib, op, self.A_test) as op:
                    flags = lib.pogs_solver_flags(direct, equil)
                    solver = lib.pogs_init(op, flags)
                    oktest.assert_noerr(lib.pogs_finish(solver, 0))

    def test_pogs_abstract_components(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            with pogs_contexts.Abstract(lib, self.A_test, obj) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_coldstart_components(ctx)

    def test_pogs_abstract_call(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            with pogs_contexts.Abstract(lib, self.A_test, obj) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_call(ctx)

    def test_pogs_abstract_call_unified(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            with pogs_contexts.Abstract(lib, self.A_test, solver=False) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_unified(ctx)

    def test_pogs_abstract_warmstart(self):
        for (lib, op, direct, equil) in self.LIBS_OPS_DIRECT_EQUIL:
            with pogs_contexts.Abstract(lib, self.A_test) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_warmstart(ctx)

    def test_pogs_abstract_io(self):
        pass
        # TODO: FIX
        equil = 1.
        for (lib, op, direct) in self.LIBS_OPS_DIRECT:
            with pogs_contexts.Abstract(lib, self.A_test) as ctx:
                self.set_standard_tolerances(ctx.lib)
                self.assert_pogs_io(ctx)