from optkit.compat import *

import numpy as np
import collections

from optkit.utils import proxutils
from optkit.tests.C.base import OptkitCTestCase
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx

NO_ERR = statements.noerr

class PogsVariablesLocal:
    def __init__(self, lib, m, n):
        self.m = m
        self.n = n
        self._state = np.zeros(6 * (m+n)).astype(lib.pyfloat)
        self._lib = lib

        for i, label in enumerate(['z', 'zt', 'z12', 'zt12', 'prev', 'temp']):
            setattr(self, label, self._state[i * (m+n) : (i+1) * (m+n)])
            if 'z' in label:
                xlabel = label.replace('z', 'x')
                ylabel = label.replace('z', 'y')
                setattr(self, ylabel, getattr(self, label)[:m])
                setattr(self, xlabel, getattr(self, label)[m:])

        self.d = np.zeros(m).astype(lib.pyfloat)
        self.e = np.zeros(n).astype(lib.pyfloat)

        def _arr2ptr(arr): return np.ravel(arr).ctypes.data_as(lib.ok_float_p)
        def _c2py(py, c): return NO_ERR(lib.vector_memcpy_av(_arr2ptr(py), c, 1))
        def load_all(solver):
            assert _c2py(self.d, solver.contents.W.contents.d)
            assert _c2py(self.e, solver.contents.W.contents.e)
            assert _c2py(self._state, solver.contents.z.contents.state)

        self.load_all_from = load_all

class PogsOutputLocal:
    def __init__(self, lib, m, n):
        self.x = np.zeros(n).astype(lib.pyfloat)
        self.y = np.zeros(m).astype(lib.pyfloat)
        self.mu = np.zeros(n).astype(lib.pyfloat)
        self.nu = np.zeros(m).astype(lib.pyfloat)
        self.ptr = lib.pogs_output(self.x.ctypes.data_as(lib.ok_float_p),
                                   self.y.ctypes.data_as(lib.ok_float_p),
                                   self.mu.ctypes.data_as(lib.ok_float_p),
                                   self.nu.ctypes.data_as(lib.ok_float_p))

def load_to_local(lib, py_vector, c_vector):
    assert NO_ERR( lib.vector_memcpy_av(
            py_vector.ctypes.data_as(lib.ok_float_p), c_vector, 1) )

class EquilibratedMatrix(object):
    def __init__(self, d, A, e):
        self.dot = lambda x: d * A.dot(e * x)
        self.shape = A.shape
        self.transpose = lambda: EquilibratedMatrix(e, A.T, d)
    @property
    def T(self):
        return self.transpose()

class Base:
    def __init__(self, libctx, m, n, obj='Abs'):
        lib = libctx.lib
        self._libctx = libctx
        self._f = okcctx.CFunctionVectorContext(lib, m)
        self._g = okcctx.CFunctionVectorContext(lib, n)
        self._objstring = obj
        self.params = collections.namedtuple(
                'TestParams',
                'shape z output info setings res tol obj f f_py g g_py')
        self.params.shape = (m, n)
        self.params.z = PogsVariablesLocal(lib, m, n)
        self.params.output = PogsOutputLocal(lib, m, n)
        self.params.info = lib.pogs_info()
        self.params.settings = settings = lib.pogs_settings()
        assert NO_ERR( lib.pogs_set_default_settings(settings) )
        self.params.settings.verbose = int(statements.VERBOSE_TEST)
        self.params.res = lib.pogs_residuals()
        self.params.tol = lib.pogs_tolerances()
        self.params.obj = lib.pogs_objective_values()

        class SolverCtx:
            def __init__(self, data, flags):
                self._data = data
                self._flags = flags
            def __enter__(self):
                self._s = lib.pogs_init(self._data, self._flags)
                return self._s
            def __exit__(self, *exc):
                assert NO_ERR( lib.pogs_finish(self._s, 0) )
        self.SolverCtx = SolverCtx

    def __enter__(self):
        m, n = self.params.shape
        lib = self.lib = self._libctx.__enter__()
        self.f = self._f.__enter__()
        self.g = self._g.__enter__()
        f, f_py, f_ptr = self.f.cptr, self.f.py, self.f.pyptr
        g, g_py, g_ptr = self.g.cptr, self.g.py, self.g.pyptr

        h = lib.function_enums.dict[self._objstring]
        asymm = 1 + int('Asymm' in self._objstring)
        for i in xrange(m):
            f_py[i] = lib.function(h, 1, 1, 1, 0, 0, asymm)
        for j in xrange(n):
            g_py[j] = lib.function(
                    lib.function_enums.IndGe0, 1, 0, 1, 0, 0, 1)
        assert NO_ERR( lib.function_vector_memcpy_va(f, f_ptr) )
        assert NO_ERR( lib.function_vector_memcpy_va(g, g_ptr) )
        self.params.f = f
        self.params.g = g
        self.params.f_py = f_py
        self.params.g_py = g_py
        return self

    def __exit__(self, *exc):
        self._f.__exit__(*exc)
        self._g.__exit__(*exc)
        self._libctx.__exit__(*exc)

class Dense(Base):
    def __init__(self, libctx, A, layout, obj='Abs'):
        m, n = A.shape
        lib = libctx.lib
        Base.__init__(self, libctx, m, n, obj=obj)
        order = 'C' if layout == lib.enums.CblasRowMajor else 'F'
        self.A_dense = np.zeros((m, n), dtype=lib.pyfloat, order=order)
        self.A_dense += A
        self.data = self.A_dense.ctypes.data_as(lib.ok_float_p)
        self.flags = lib.pogs_solver_flags(m, n, layout)

    def __enter__(self):
        Base.__enter__(self)
        self.solver = self.SolverCtx(self.data, self.flags)
        return self

    def __exit__(self, *exc):
        Base.__exit__(self, *exc)
        del self.solver

class Abstract(Base):
    def __init__(self, lib, A, optype, direct, equil, obj='Abs', solver=True):
        m, n = A.shape
        Base.__init__(self, lib, m, n, obj=obj)
        self.optype = optype
        self.direct = direct
        self.equil = equil

        rowmajor = A.flags.c_contiguous
        self.A_dense = A
        self._op = okcctx.c_operator_context(lib, optype, A, rowmajor)
        self.data = None
        self.flags = lib.pogs_solver_flags(direct, equil)

    def __enter__(self):
        Base.__enter__(self)
        self.op = self.data = self._op.__enter__()
        self.solver = self.SolverCtx(self.data, self.flags)
        return self

    def __exit__(self):
        if self._solver:
            oktest.assert_noerr(self.lib.pogs_free(self.solver))
        self._op.__exit__()
        Base.__exit__(self)
