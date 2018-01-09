from optkit.compat import *

import os
import numpy as np
import ctypes as ct

from optkit.types.operator import OperatorTypes
from optkit.types.pogs.base import PogsTypesBase

class PogsAbstractTypes(PogsTypesBase):
    def __init__(self, backend):
        lib = backend.pogs_abstract
        PogsTypesBase.__init__(self, lib)

        _SolverBase = self._SolverBase
        _SolverCacheBase = self._SolverCacheBase

        Operator = OperatorTypes(backend, lib).AbstractLinearOperator

        class SolverCache(_SolverCacheBase):
            def __init__(self, m, n, cache=None):
                shapes = {
                        'd': m,
                        'e': n,
                }
                _SolverCacheBase.__init__(self, shapes, cache)
                # TODO: OTHER PROCESS OPERATOR

        class Solver(_SolverBase):
            def __init__(self, A, **options):
                if not isinstance(A, np.ndarray) or len(A.shape) != 2:
                    raise TypeError('input must be a 2-d {}'.format(np.ndarray))
                _SolverBase.__init__(self, A, **options)

            def _unregister_solver(self):
                _SolverBase._unregister_solver(self)
                if self.A is not None:
                    self.A.release_operator()

            @property
            def A(self):
                try:
                    return self._A
                except:
                    return None

            @A.setter
            def A(self, A):
                if self.A is not None:
                    raise AttributeError('solver.A is set-once attribute')
                self._A = Operator(A)

            def _build_solver_data(self, A):
                return A.c_ptr

            def _build_solver_flags(self, A, **options):
                direct = int(options.pop('direct', False))
                equil_norm = float(options.pop('equil_norm', 1.))
                return lib.pogs_solver_flags(direct, equil_norm)

            def _solver_cache_from_dict(self, cache, **options):
                err = (
                    'd' not in cache or
                    'e' not in cache
                )
                # TODO: OPERATOR IO

                solver_cache = SolverCache(self.m, self.n, cache)

                if err:
                    raise ValueError(
                            'Minimal requirements to load solver '
                            'not met. Specified file(s) must contain '
                            'at least one .npz file with entries '
                            '"d" and "e", or the specified folder '
                            'must contain .npy files of the same names.')

            def _allocate_solver_cache(self):
                return SolverCache(lib, self.m, self.n)

        self.Solver = Solver