from optkit.compat import *

import os
import numpy as np
import ctypes as ct

from optkit.types.operator import OperatorTypes
from optkit.types.pogs.common import PogsCommonTypes

class PogsAbstractTypes(PogsCommonTypes):
    def __init__(self, backend):
        lib = backend.pogs_abstract
        PogsCommonTypes.__init__(self, lib)
        Objective = self.Objective
        SolverSettings = self.SolverSettings
        SolverInfo = self.SolverInfo
        SolverOutput = self.SolverOutput

        Operator = OperatorTypes(backend, lib).AbstractLinearOperator

        class Solver(object):
            def __del__(self):
                self.__unregister_solver()
                self.A.release_operator()

            def __init__(self, A, **options):
                self.__A = Operator(A)
                self.shape = (self.m, self.n) = (m, n) = self.A.shape
                self.__f = np.zeros(m).astype(lib.function)
                self.__f_ptr = self.__f.ctypes.data_as(lib.function_p)
                self.__f_c = lib.function_vector(self.m, self.__f_ptr)
                self.__g = np.zeros(n).astype(lib.function)
                self.__g_ptr = self.__g.ctypes.data_as(lib.function_p)
                self.__g_c = lib.function_vector(self.n, self.__g_ptr)
                self.__c_solver = None

                NO_INIT = bool(options.pop('no_init', False))
                DIRECT = int(options.pop('direct', False))
                EQUILNORM = float(options.pop('equil_norm', 1.))

                if not NO_INIT:
                    self.__register_solver(lib.pogs_init(
                            self.A.c_ptr, DIRECT, EQUILNORM))

                self.settings = SolverSettings()
                self.info = SolverInfo()
                self.output = SolverOutput(m, n)
                self.settings.update(**options)
                self.first_run = True

            @property
            def A(self):
                return self.__A

            @property
            def c_solver(self):
                return self.__c_solver

            def __register_solver(self, solver):
                self.__c_solver = solver
                self.exit_call = lib.pogs_finish

            def __unregister_solver(self):
                if self.c_solver is None:
                    return
                self.exit_call(self.c_solver, 0)
                self.__c_solver = None
                self.exit_call = lambda arg1, arg2: None

            def __update_function_vectors(self, f, g):
                for i in xrange(f.size):
                    self.__f[i] = lib.function(
                            f.h[i], f.a[i], f.b[i], f.c[i], f.d[i], f.e[i],
                            f.s[i])

                for j in xrange(g.size):
                    self.__g[j] = lib.function(
                            g.h[j], g.a[j], g.b[j], g.c[j], g.d[j], g.e[j],
                            g.s[j])

            def solve(self, f, g, **options):
                if self.c_solver is None:
                    raise ValueError(
                            'No solver intialized, solve() call invalid')

                if not isinstance(f, Objective) and isinstance(g, Objective):
                    raise TypeError(
                        'inputs f, g must be of type {} \nprovided: {}, '
                        '{}'.format(Objective, type(f), type(g)))

                if not (f.size == self.m and g.size == self.n):
                    raise ValueError(
                        'inputs f, g not compatibly sized with solver'
                        '\nsolver dimensions ({}, {})\n provided: '
                        '({}{})'.format(self.m, self.n, f.size, g.size))


#               # TODO : logic around resume, warmstart, rho input
                self.__update_function_vectors(f, g)
                self.settings.update(**options)
                lib.pogs_solve(
                        self.c_solver, self.__f_c, self.__g_c, self.settings.c,
                        self.info.c, self.output.c)
                self.first_run = False

        self.Solver = Solver