from optkit.compat import *

import unittest
import numpy as np

import optkit
from optkit.api import backend
from optkit import set_backend

def NO_ERR(call): return call == 0

class BackendTestCase(unittest.TestCase):
    @staticmethod
    def object_blocks_reset(constructor):
        with constructor() as obj:
            assert obj is not None
            assert not backend.device_reset_allowed
        assert backend.device_reset_allowed
        assert NO_ERR(backend.reset_device())
        return True

    def test_libs(self):
        pogs_libnames = []
        pogs_abstract_libnames = []
        cluster_libnames = []

        assert ( backend.pogs_lib_loader is not None )
        assert ( backend.pogs_abstract_lib_loader is not None )
        assert ( backend.cluster_lib_loader is not None )

        for libkey in backend.pogs_lib_loader.libs:
            if backend.pogs_lib_loader.libs[libkey] is not None:
                pogs_libnames.append(libkey)

        for libkey in backend.pogs_lib_loader.libs:
            if backend.pogs_abstract_lib_loader.libs[libkey] is not None:
                pogs_abstract_libnames.append(libkey)

        for libkey in backend.cluster_lib_loader.libs:
            if backend.cluster_lib_loader.libs[libkey] is not None:
                cluster_libnames.append(libkey)

        print('available libs, POGS:', pogs_libnames)
        assert pogs_libnames

        print('available libs, POGS abstract:', pogs_abstract_libnames)
        assert pogs_abstract_libnames

        print('avaialable libs, clustering:', cluster_libnames)
        assert cluster_libnames

        configs = []
        for libname in pogs_libnames:
            gpu = 'gpu' in libname
            double = '64' in libname

            assert NO_ERR(backend.reset_device())
            assert NO_ERR(set_backend(gpu=gpu, double=double))

            # assert not None
            assert backend.pogs
            assert backend.pogs_abstract
            assert backend.cluster

            assert ( backend.device_is_gpu == gpu )
            assert ( backend.precision_is_32bit != double )

            def build_pogs():
                return optkit.api.PogsSolver(np.random.rand(10, 5))

            def build_apogs():
                return optkit.api.PogsAbstractSolver(np.random.rand(10, 5))

            def build_clus():
                c = optkit.api.Clustering()
                c.kmeans_work_init(10, 5, 5)
                return c

            assert self.object_blocks_reset(build_pogs)
            assert self.object_blocks_reset(build_apogs)
            assert self.object_blocks_reset(build_clus)