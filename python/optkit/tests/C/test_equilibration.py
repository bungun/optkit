from optkit.compat import *

import os
import numpy as np
import itertools
import unittest

from optkit.libs.equilibration import EquilibrationLibs
from optkit.libs import enums
from optkit.tests import defs
from optkit.tests.C import context_managers as okcctx
from optkit.tests.C import statements

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances
CUSTOM_TOLS = statements.custom_tolerances

def equilibrate(lib, order, A_test):
    m, n = A_test.shape
    RTOL, ATOLM, _, _ = CUSTOM_TOLS(lib, m, n, 2, 2, 7)

    hdl = okcctx.CDenseLinalgContext(lib)
    A = okcctx.CDenseMatrixContext(lib, m, n, order, value=A_test)
    d = okcctx.CVectorContext(lib, m)
    e = okcctx.CVectorContext(lib, n)

    x_test = np.random.random(n)
    A_in_py = A_test.astype(lib.pyfloat)
    A_in_ptr = A_in_py.ctypes.data_as(lib.ok_float_p)
    order_in = lib.enums.CblasRowMajor if A_in_py.flags.c_contiguous else \
               lib.enums.CblasColMajor

    with A, d, e, hdl as hdl:
        assert NO_ERR( lib.regularized_sinkhorn_knopp(
                hdl, A_in_ptr, A.c, d.c, e.c, order_in) )
        A.sync_to_py()
        d.sync_to_py()
        e.sync_to_py()

        A_eqx = np.dot(A.py, x_test)
        DAEx = d.py * np.dot(A_in_py, (e.py * x_test))
        assert VEC_EQ( A_eqx, DAEx, ATOLM, RTOL )

    return True

class EquilLibsTestCase(unittest.TestCase):
    """
        Equilibrate input A_in as

            D * A_equil * E

        with D, E, diagonal.

        Test that

            D * A_equil * E == A_in,

        or that

            D^-1 * A_in * E^-1 == A_equil.
    """
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(EquilibrationLibs())
        self.A_test = dict(
                dense=defs.A_test_gen(),
                sparse=defs.A_test_sparse_gen())

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.x_test = np.random.random(defs.shape()[1])
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)
        self.LIBS_OPS = itertools.product(self.libs, defs.OPKEYS)

    def test_libs_exist(self):
        assert any(self.libs)

    def test_regularized_sinkhorn_knopp(self):
        A_test = self.A_test['dense']
        m, n = A_test.shape
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                print('regularized sinkhorn, CBLAS layout:', order)

                assert equilibrate(lib, order, A_test)

                A_rowmissing = np.zeros_like(A_test)
                A_rowmissing += A_test
                A_rowmissing[int(m/2), :] *= 0
                assert equilibrate(lib, order, A_rowmissing)

                A_colmissing = np.zeros_like(A_test)
                A_colmissing += A_test
                A_colmissing[:, int(n/2)] *= 0
                assert equilibrate(lib, order, A_colmissing)

    def test_operator_sinkhorn_knopp(self):
        """test equilibration for each operator type defined in defs.OPKEYS"""
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL, _, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 2, 2, 7)

                print('operator sinkhorn, operator type:', op_)
                hdl = okcctx.CDenseLinalgContext(lib)
                A = self.A_test[op_]
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m)
                d = okcctx.CVectorContext(lib, m)
                e = okcctx.CVectorContext(lib, n)
                o = okcctx.c_operator_context(lib, op_, A)

                with x, y, d, e, o as o, hdl as hdl:
                    # equilibrate operator
                    assert NO_ERR( lib.operator_regularized_sinkhorn(
                            hdl, o, d.c, e.c, 1.) )

                    # extract results
                    d.sync_to_py()
                    e.sync_to_py()
                    DAEx = d.py * A.dot(e.py * x.py)

                    assert NO_ERR( o.contents.apply(o.contents.data, x.c, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( y.py, DAEx, ATOLN, RTOL )

    def test_operator_equil(self):
        """ test equilibration for each operator type defined in defs.OPKEYS
        """
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL, _, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 2, 2, 7)
                print('operator equil generic operator type:', op_)

                hdl = okcctx.CDenseLinalgContext(lib)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m)
                d = okcctx.CVectorContext(lib, m)
                e = okcctx.CVectorContext(lib, n)
                A = self.A_test[op_]
                o = okcctx.c_operator_context(lib, op_, A)

                with x, y, d, e, o as o, hdl as hdl:
                    status = lib.operator_equilibrate(hdl, o, d.c, e.c, 1.)

                    d.sync_to_py()
                    e.sync_to_py()
                    DAEx = d.py * np.dot(A, (e.py * x.py))

                    o.contents.apply(o.contents.data, x.c, y.c)
                    y.sync_to_py()

                    # TODO: FIX THIS TEST
                    # assert( status == 0 )
                    # assert VEC_EQ( y.py, DAEx, ATOLN, RTOL )

    def test_operator_norm(self):
        """ test norm estimation for each operator type defined in defs.OPKEYS
        """
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL = 5e-2
                ATOL = 5e-3 * (m*n)**0.5

                print('operator norm, operator type:', op_)
                hdl = okcctx.CDenseLinalgContext(lib)
                A = self.A_test[op_]
                o = okcctx.c_operator_context(lib, op_, A)

                # estimate operator norm
                normest_p = lib.ok_float_p()
                normest_p.contents = lib.ok_float(0.)
                pynorm = np.linalg.norm(A)

                with o as o, hdl as hdl:
                    assert NO_ERR(lib.operator_estimate_norm(hdl, o, normest_p))
                    cnorm = normest_p.contents

                    if defs.VERBOSE_TEST:
                        print('operator norm, Python: ', pynorm)
                        print('norm estimate, C: ', cnorm)

                    assert( cnorm >= ATOL + RTOL * pynorm
                            or pynorm >= ATOL + RTOL * cnorm )
