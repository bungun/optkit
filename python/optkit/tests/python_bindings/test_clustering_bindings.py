# import unittest
# import numpy as np
# import gc
# import os
# from subprocess import call
# from os import path
# from optkit import *
# from optkit.api import backend
# from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH

# class ClusteringBindingsTestCase(unittest.TestCase):
# 	@classmethod
# 	def setUpClass(self):
# 		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
# 		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'

# 	@classmethod
# 	def tearDownClass(self):
# 		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

# 	def setUp(self):
# 		self.shape = None
# 		if DEFAULT_MATRIX_PATH is not None:
# 			try:
# 				self.A_test = np.load(DEFAULT_MATRIX_PATH)
# 				self.shape = A.shape
# 			except:
# 				pass
# 		if self.shape is None:
# 			self.shape = DEFAULT_SHAPE
# 			self.A_test = np.random.rand(*self.shape)