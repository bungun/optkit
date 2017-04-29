from optkit.compat import *

import unittest
import numpy as np

import optkit
from optkit.api import backend
from optkit import set_backend

class BackendTestCase(unittest.TestCase):
	def test_libs(self):
		pogs_libnames = []
		pogs_abstract_libnames = []
		cluster_libnames = []

		self.assertIsNotNone( backend.pogs_lib_loader )
		self.assertIsNotNone( backend.pogs_abstract_lib_loader )
		self.assertIsNotNone( backend.cluster_lib_loader )

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
		self.assertTrue( len(pogs_libnames) > 0 )

		print('available libs, POGS abstract:', pogs_abstract_libnames)
		self.assertTrue( len(pogs_abstract_libnames) > 0 )

		print('avaialable libs, clustering:', cluster_libnames)
		self.assertTrue( len(cluster_libnames) > 0 )

		configs = []
		for libname in pogs_libnames:
			gpu = 'gpu' in libname
			double = '64' in libname

			self.assertEqual( backend.reset_device(), 0 )
			self.assertEqual( set_backend(gpu=gpu, double=double), 0 )

			self.assertIsNotNone( backend.pogs )
			self.assertIsNotNone( backend.pogs_abstract )
			self.assertIsNotNone( backend.cluster )

			self.assertEqual( backend.device_is_gpu, gpu )
			self.assertNotEqual( backend.precision_is_32bit, double )

			s = optkit.api.PogsSolver(np.random.rand(10, 5))
			self.assertFalse( backend.device_reset_allowed )
			del s
			self.assertTrue( backend.device_reset_allowed )
			backend.reset_device()

			# TODO: add solver initialization/deletion for PogsAbstract?

			c = optkit.api.Clustering()
			c.kmeans_work_init(10, 5, 5)
			self.assertFalse( backend.device_reset_allowed )
			del c
			self.assertTrue( backend.device_reset_allowed )
			backend.reset_device()