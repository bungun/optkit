from optkit.compat import *

import numpy as np
import collections

from optkit.utils import proxutils
from optkit.tests.C.base import OptkitCTestCase
import optkit.tests.C.statements as okctest
import optkit.tests.C.context_managers as okcctx


class PogsVariablesLocal():
    def __init__(self, m, n, pytype):
        self.m = m
        self.n = n
        self.z = np.zeros(m + n).astype(pytype)
        self.z12 = np.zeros(m + n).astype(pytype)
        self.zt = np.zeros(m + n).astype(pytype)
        self.zt12 = np.zeros(m + n).astype(pytype)
        self.prev = np.zeros(m + n).astype(pytype)
        self.d = np.zeros(m).astype(pytype)
        self.e = np.zeros(n).astype(pytype)

    @property
    def x(self):
        return self.z[self.m:]

    @property
    def y(self):
        return self.z[:self.m]

    @property
    def x12(self):
        return self.z12[self.m:]

    @property
    def y12(self):
        return self.z12[:self.m]

    @property
    def xt(self):
        return self.zt[self.m:]

    @property
    def yt(self):
        return self.zt[:self.m]

    @property
    def xt12(self):
        return self.zt12[self.m:]

    @property
    def yt12(self):
        return self.zt12[:self.m]

class PogsOutputLocal():
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
    assert okctest.noerr( lib.vector_memcpy_av(
            py_vector.ctypes.data_as(lib.ok_float_p), c_vector, 1) )

def load_all_local(lib, py_vars, solver):
    if not isinstance(py_vars, PogsVariablesLocal):
        raise TypeError('argument "py_vars" must be of type {}'.format(
                        PogsVariablesLocal))
    elif 'pogs_solver_p' not in lib.__dict__:
        raise ValueError('argument "lib" must contain field named '
                         '"pogs_solver_p"')
    elif not isinstance(solver, lib.pogs_solver_p):
        raise TypeError('argument "solver" must be of type {}'.format(
                        lib.pogs_solver_p))

    z = solver.contents.z
    W = solver.contents.W
    load_to_local(lib, py_vars.z, z.contents.primal.contents.vec)
    load_to_local(lib, py_vars.z12, z.contents.primal12.contents.vec)
    load_to_local(lib, py_vars.zt, z.contents.dual.contents.vec)
    load_to_local(lib, py_vars.zt12, z.contents.dual12.contents.vec)
    load_to_local(lib, py_vars.prev, z.contents.prev.contents.vec)
    load_to_local(lib, py_vars.d, W.contents.d)
    load_to_local(lib, py_vars.e, W.contents.e)

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
        self.params.z = PogsVariablesLocal(m, n, lib.pyfloat)
        self.params.output = PogsOutputLocal(lib, m, n)
        self.params.info = lib.pogs_info()
        self.params.settings = settings = lib.pogs_settings()
        assert okctest.noerr( lib.pogs_set_default_settings(settings) )
        self.params.settings.verbose = int(okctest.VERBOSE_TEST)
        self.params.res = lib.pogs_residuals()
        self.params.tol = lib.pogs_tolerances()
        self.params.obj = lib.pogs_objective_values()

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
        assert okctest.noerr( lib.function_vector_memcpy_va(f, f_ptr) )
        assert okctest.noerr( lib.function_vector_memcpy_va(g, g_ptr) )
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
    def __init__(self, libctx, A, layout, obj='Abs', solver=True):
        m, n = A.shape
        lib = libctx.lib
        Base.__init__(self, libctx, m, n, obj=obj)
        order = 'C' if layout == lib.enums.CblasRowMajor else 'F'
        self.A_dense = np.zeros((m, n), dtype=lib.pyfloat, order=order)
        self.A_dense += A
        self.data = self.A_dense.ctypes.data_as(lib.ok_float_p)
        self.flags = lib.pogs_solver_flags(m, n, layout)
        self._solver = solver

    def __enter__(self):
        Base.__enter__(self)
        if self._solver:
            self.solver = self.lib.pogs_init(self.data, self.flags)
        return self

    def __exit__(self, *exc):
        if self._solver:
            assert okctest.noerr(self.lib.pogs_finish(self.solver, 0))
        Base.__exit__(self, *exc)

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
        self._solver = solver

    def __enter__(self):
        Base.__enter__(self)
        self.op = self.data = self._op.__enter__()
        if self._solver:
            self.solver = self.lib.pogs_init(self.data, self.flags)
        return self

    def __exit__(self):
        if self._solver:
            oktest.assert_noerr(self.lib.pogs_free(self.solver))
        self._op.__exit__()
        Base.__exit__(self)
