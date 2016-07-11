import unittest
import numpy as np
import optkit
from optkit.api import backend
from optkit import set_backend
from optkit.compat import *

class BackendTestCase(unittest.TestCase):
	def test_libs(self):
		pogs_libnames = []
		cluster_libnames = []

		self.assertNotEqual(backend.pogs_lib_loader, None)
		self.assertNotEqual(backend.cluster_lib_loader, None)

		for libkey in backend.pogs_lib_loader.libs:
			if backend.pogs_lib_loader.libs[libkey] is not None:
				pogs_libnames.append(libkey)

		for libkey in backend.cluster_lib_loader.libs:
			if backend.cluster_lib_loader.libs[libkey] is not None:
				cluster_libnames.append(libkey)

		print('available libs, POGS:', pogs_libnames)
		self.assertTrue( len(pogs_libnames) > 0 )

		print('avaialable libs, clustering:', cluster_libnames)
		self.assertTrue( len(cluster_libnames) > 0 )

		configs = []
		for libname in pogs_libnames:
			gpu = 'gpu' in libname
			double = '64' in libname

			self.assertEqual( backend.reset_device(), 0 )
			self.assertEqual( set_backend(gpu=gpu, double=double), 0 )

			self.assertNotEqual( backend.pogs, None )
			self.assertNotEqual( backend.cluster, None )

			self.assertEqual( backend.device_is_gpu, gpu )
			self.assertNotEqual( backend.precision_is_32bit, double )

			s = optkit.api.PogsSolver(np.random.rand(10, 5))
			self.assertFalse( backend.device_reset_allowed )
			del s
			self.assertTrue( backend.device_reset_allowed )
			backend.reset_device()

			c = optkit.api.Clustering()
			c.kmeans_work_init(10, 5, 5)
			self.assertFalse( backend.device_reset_allowed )
			del c
			self.assertTrue( backend.device_reset_allowed )
			backend.reset_device()

