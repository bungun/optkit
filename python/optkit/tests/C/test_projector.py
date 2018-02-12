from optkit.compat import *

import os
import numpy as np
import itertools
import unittest

from optkit.libs.projector import ProjectorLibs
from optkit.libs import enums
from optkit.tests import defs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances
CUSTOM_TOLS = statements.custom_tolerances

class DirectProjectorTestCase(unittest.TestCase):
    """
    TODO: docstring
    """
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(ProjectorLibs())
        self.A_test = defs.A_test_gen()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)

    def test_libs_exist(self):
        assert any(self.libs)

    def test_projection(self):
        """projection test

            (1a) generate random A, x, y
            (1b) optionally, normalize A:
            (2) project (x, y) onto graph y = Ax

            Given matrix A \in R^{m x n}, and input vectors y \in R^m
            and x \in R^n:


            (0) optionally, normalize A. divide all entries of A by

                 \sum_{i=1}^m {a_i'a_i} / \sqrt{m},  if m >= n
                 \sum_{j=1}^n {a_j'a_j} / \sqrt{n},  otherwise

            (1) calculate Cholesky factorization of

                (I + AA') if m >= n or
                (I + A'A) otherwise

            (2) set (x_out, y_out) = Proj_{y = Ax} (x, y)


            The equality

                y_out == A * x_out

            should hold elementwise to float/double precision
        """
        m, n = defs.shape()
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)
                for normalize in (False, True):
                    x_in = okcctx.CVectorContext(lib, n, random=True)
                    y_in = okcctx.CVectorContext(lib, m, random=True)
                    x_out = okcctx.CVectorContext(lib, n)
                    y_out = okcctx.CVectorContext(lib, m)
                    A = okcctx.CDenseMatrixContext(lib, m, n, order, random=True)
                    hdl = okcctx.CDenseLinalgContext(lib)
                    lapack_hdl = okcctx.CDenseLapackContext(lib)

                    skinny = 1 if m >= n else 0

                    with x_in, y_in, x_out, y_out, A, hdl as hdl:

                        P = lib.direct_projector(None, None, 0, skinny, 0)
                        def build_(): return lib.direct_projector_alloc(P, A.c)
                        def free_(): return lib.direct_projector_free(P)
                        with okcctx.CVariableContext(build_, free_):
                            assert NO_ERR( lib.direct_projector_initialize(
                                    hdl, lapack_hdl, P, normalize) )
                            assert NO_ERR( lib.direct_projector_project(
                                    hdl, lapack_hdl, P, x_in.c, y_in.c, x_out.c,
                                    y_out.c) )

                            # copy results
                            x_out.sync_to_py()
                            y_out.sync_to_py()

                            # test projection y_out == Ax_out
                            if normalize:
                                Ax = np.dot(A.py, x_out.py) / P.normA
                            else:
                                Ax = np.dot(A.py, x_out.py)
                            assert VEC_EQ( Ax, y_out.py, ATOLM, RTOL )

class IndirectProjectorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(ProjectorLibs())
        self.A_test = dict(
                dense=defs.A_test_gen(),
                sparse=defs.A_test_sparse_gen())

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_OPS = itertools.product(self.libs, defs.OPKEYS)

    def test_alloc_free(self):
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                if defs.VERBOSE_TEST:
                    print('test indirect projector alloc, operator type:', op_)

                A = okcctx.c_operator_context(lib, op_, self.A_test[op_])
                with A as A:
                    p = lib.indirect_projector(None, None)
                    try:
                        assert NO_ERR( lib.indirect_projector_alloc(p, A) )
                        assert ( p.A != 0 )
                        assert ( p.cgls_work != 0 )
                    finally:
                        assert NO_ERR( lib.indirect_projector_free(p) )

    def test_projection(self):
        """ test projection for each operator type """
        m, n = defs.shape()
        for lib, op_ in itertools.product(self.libs, defs.OPKEYS):
            with lib as lib:
                RTOL, ATOLM, _, _ = CUSTOM_TOLS(lib, m, n, 1)
                if defs.VERBOSE_TEST:
                    print('indirect projection, operator type:', op_)

                x_in = okcctx.CVectorContext(lib, n, random=True)
                y_in = okcctx.CVectorContext(lib, m, random=True)
                x_out = okcctx.CVectorContext(lib, n)
                y_out = okcctx.CVectorContext(lib, m)
                hdl = okcctx.CDenseLinalgContext(lib)
                A = self.A_test[op_]
                o = okcctx.c_operator_context(lib, op_, A)

                with x_in, y_in, x_out, y_out, o as o, hdl as hdl:
                    p = lib.indirect_projector(None, None, 0)
                    def build_(): return lib.indirect_projector_alloc(p, o)
                    def free_(): return lib.indirect_projector_free(p)
                    with okcctx.CVariableContext(build_, free_):
                        assert NO_ERR( lib.indirect_projector_project(
                                hdl, p, x_in.c, y_in.c, x_out.c, y_out.c) )

                        x_out.sync_to_py()
                        y_out.sync_to_py()
                        assert VEC_EQ( A.dot(x_out.py), y_out.py, ATOLM, RTOL )


class DenseDirectProjectorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(ProjectorLibs())
        self.A_test = defs.A_test_gen()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)

    def test_alloc_free(self):
        m, n = defs.shape()
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                with A:
                    p = None
                    try:
                        p = lib.dense_direct_projector_alloc(A.c)
                        assert ( p.contents.kind == lib.enums.DENSE_DIRECT)
                        assert ( p.contents.size1 == m)
                        assert ( p.contents.size2 == n)
                        assert (p.contents.data != 0)
                        assert (p.contents.initialize != 0)
                        assert (p.contents.project != 0)
                        assert (p.contents.free != 0)
                    finally:
                        assert NO_ERR( p.contents.free(p.contents.data) )

    def test_projection(self):
        m, n = defs.shape()
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                TOL_PLACEHOLDER = 1e-8
                RTOL, ATOLM, ATOLN, ATOLMN = CUSTOM_TOLS(lib, m, n, 1, 3, 7)

                x_in = okcctx.CVectorContext(lib, n, random=True)
                y_in = okcctx.CVectorContext(lib, m, random=True)
                x_out = okcctx.CVectorContext(lib, n)
                y_out = okcctx.CVectorContext(lib, m)
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)

                with x_in, y_in, x_out, y_out, A as A:
                    def build_(): return lib.dense_direct_projector_alloc(A.c)
                    with okcctx.CVariableAutoContext(build_) as p:
                        assert NO_ERR( p.contents.initialize(p.contents.data, 0) )
                        assert NO_ERR( p.contents.project(
                                p.contents.data, x_in.c, y_in.c, x_out.c,
                                y_out.c, TOL_PLACEHOLDER) )

                        x_out.sync_to_py()
                        y_out.sync_to_py()
                        assert VEC_EQ( A.py.dot(x_out.py), y_out.py, ATOLM, RTOL )


class GenericIndirectProjectorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(ProjectorLibs())
        self.A_test = dict(
                dense=defs.A_test_gen(),
                sparse=defs.A_test_sparse_gen())

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_OPS = itertools.product(self.libs, defs.OPKEYS)

    def test_alloc_free(self):
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                if defs.VERBOSE_TEST:
                    print('test indirect projector alloc, operator type:', op_)

                A = okcctx.c_operator_context(lib, op_, self.A_test[op_])
                with A as A:
                    p = None
                    try:
                        p = lib.indirect_projector_generic_alloc(A)
                        assert ( p.contents.kind == lib.enums.INDIRECT )
                        assert ( p.contents.size1 == m )
                        assert ( p.contents.size2 == n )
                        assert ( p.contents.data != 0 )
                        assert ( p.contents.initialize != 0 )
                        assert ( p.contents.project != 0 )
                        assert ( p.contents.free != 0 )
                    finally:
                        assert NO_ERR(p.contents.free(p.contents.data))

    def test_projection(self):
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                TOL_CG = 1e-12
                RTOL, ATOLM, _, _ = CUSTOM_TOLS(lib, m, n, 1, 2, 7)
                if defs.VERBOSE_TEST:
                    print('indirect projection, operator type:', op_)


                x_in = okcctx.CVectorContext(lib, n, random=True)
                y_in = okcctx.CVectorContext(lib, m, random=True)
                x_out = okcctx.CVectorContext(lib, n)
                y_out = okcctx.CVectorContext(lib, m)
                A = self.A_test[op_]
                o = okcctx.c_operator_context(lib, op_, self.A_test[op_])

                with x_in, y_in, x_out, y_out, o as o:
                    def build_(): return lib.indirect_projector_generic_alloc(o)
                    with okcctx.CVariableAutoContext(build_) as p:
                        assert NO_ERR( p.contents.project(
                                p.contents.data, x_in.c, y_in.c, x_out.c,
                                y_out.c, TOL_CG) )

                        x_out.sync_to_py()
                        y_out.sync_to_py()
                        assert VEC_EQ( A.dot(x_out.py), y_out.py, ATOLM, RTOL )
