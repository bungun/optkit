from optkit.compat import *

import os
import numpy as np
import numpy.linalg as la

from optkit.types.pogs.base import PogsTypesBase

class PogsDenseDirectTypes(PogsTypesBase):
    def __init__(self, backend):
        lib = backend.pogs
        PogsTypesBase.__init__(self, lib)

        _SolverBase = self._SolverBase
        _SolverCacheBase = self._SolverCacheBase

        class SolverCache(_SolverCacheBase):
            def __init__(self, m, n, cache=None):
                mindim = min(m, n)
                shapes = {
                        'A_equil': (m, n),
                        'd': m,
                        'e': n,
                        'ATA_cholesky': (mindim, mindim)
                }
                _SolverCacheBase.__init__(self, shapes, cache)

                A_rowmajor = self.A_equil.flags.c_contiguous
                L_rowmajor = self.ATA_cholesky.flags.c_contiguous
                if (A_rowmajor != L_rowmajor):
                    raise ValueError('cached matrix and cholesky '
                                     'factorization have different '
                                     'data layouts')

                self.flags.m, self.flags.n = m, n
                self.flags.layout = (
                        lib.enums.CblasRowMajor if A_rowmajor else
                        lib.enums.CblasColMajor)

        class Solver(_SolverBase):
            def __init__(self, A, **options):
                if not isinstance(A, np.ndarray) or len(A.shape) != 2:
                    raise TypeError('input must be a 2-d {}'.format(np.ndarray))
                _SolverBase.__init__(self, A, **options)

            @A.setter
            def A(self, A):
                if self.A is not None:
                    raise AttributeError('solver.A is set-once attribute')
                self._A = A.astype(lib.pyfloat)

            def _build_solver_data(self, A):
                if A.dtype.type is not lib.pyfloat:
                    raise ValueError('input matrix must be of type {}'
                                     ''.format(lib.pyfloat))
                return A.ctypes.data_as(lib.ok_float_p)

            def _build_solver_flags(self, A, **options):
                layout = (
                        lib.enums.CblasRowMajor if A.flags.c_contiguous else
                        lib.enums.CblasColMajor)
                return lib.pogs_solver_flags(m, n, layout)

            def _solver_cache_from_dict(self, cache, allow_cholesky=False, **options):
                err = (
                    'A_equil' not in cache or
                    'd' not in cache or
                    'e' not in cache
                )

                solver_cache = SolverCache(self.m, self.n, cache)

                if not err and 'LLT' not in cache:
                    if allow_cholesky:
                        if self.m >= self.n:
                            AA = A_equil.T.dot(A_equil)
                        else:
                            AA = A_equil.dot(A_equil.T)
                        mean_diag = 0
                        for i in xrange(mindim):
                            mean_diag += AA[i, i]

                        mean_diag /= AA.shape[0]
                        normA = sqrt()
                        solver_cache.ATA_cholesky = la.cholesky(AA)
                    else:
                        err = 1

                if err:
                    raise ValueError(
                            'Minimal requirements to load solver '
                            'not met. Specified file(s) must contain '
                            'at least one .npz file with entries "A_equil",'
                            ' "LLT", "d", and "e", or the specified folder '
                            'must contain .npy files of the same names.')

            def _allocate_solver_cache(self):
                return SolverCache(lib, self.m, self.n)

        self.Solver = Solver