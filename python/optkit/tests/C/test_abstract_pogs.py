import unittest
import os
import numpy as np
from ctypes import c_void_p, byref, cast, addressof
from optkit.libs import DenseLinsysLibs, ProxLibs, PogsAbstractLibs
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH, \
							  significant_digits

ALPHA_DEFAULT = 1.7
RHO_DEFAULT = 1
MAXITER_DEFAULT = 2000
ABSTOL_DEFAULT = 1e-4
RELTOL_DEFAULT = 1e-3
ADAPTIVE_DEFAULT = 1
GAPSTOP_DEFAULT = 0
WARMSTART_DEFAULT = 0
VERBOSE_DEFAULT = 2
SUPPRESS_DEFAULT = 0
RESUME_DEFAULT = 0

class PogsAbstractLibsTestCase(unittest.TestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.prox_libs = ProxLibs()
		self.pogs_libs = PogsAbstractLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		dlibs = []
		pxlibs = []
		libs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(
					single_precision=single_precision, gpu=gpu))
			pxlibs.append(self.prox_libs.get(
					dlibs[-1], single_precision=single_precision, gpu=gpu))
			libs.append(self.pogs_libs.get(
					dlibs[-1], pxlibs[-1], single_precision=single_precision,
					gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(pxlibs))
		self.assertTrue(any(libs))