from optkit.compat import *

import gc
import os
import numpy as np

from optkit import *
from optkit.api import backend
from optkit.tests.defs import OptkitTestCase, TEST_ITERATE

class PogsBindingsTestCase(OptkitTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.__files = []

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig
		for f in self.__files:
			if os.path.exists(f):
				os.remove(f)

	def setUp(self):
		self.A_test = self.A_test_gen

	def register_file(self, file):
		if os.path.exists(file):
			self.__files.append(file)

	def test_objective(self):
		m = self.shape[0]
		f = PogsObjective(m, h='Abs', b=1, d=2, e=3)
		h, a, b, c, d, e = f.arrays

		for i in xrange(m):
			self.assertEqual(h[i], f.enums.dict['Abs'])
			self.assertAlmostEqual(a[i], 1)
			self.assertAlmostEqual(b[i], 1)
			self.assertAlmostEqual(c[i], 1)
			self.assertAlmostEqual(d[i], 2)
			self.assertAlmostEqual(e[i], 3)

		# set block by scalar
		f.set(end=int(m / 2), b=0.5)
		for i in xrange(int(m / 2)):
			self.assertEqual(f.b[i], 0.5)

		# set block by vector
		c = np.random.rand(int(m / 4))
		f.set(start=int(m / 2), end=int(m / 2) + int(m / 4), c=c)
		for i, idx in enumerate(
				xrange(int(m / 2), int(m / 2) + int(m / 4))):
			self.assertAlmostEqual(f.c[idx], c[i])

		# set indices by scalar
		indices = []
		threshold = lambda i: i if np.random.rand() > 0.75 else False
		while len(indices) == 0:
			indices = listfilter(None, listmap(threshold, xrange(m)))
		f.set(range=indices, e=0.1)
		for idx in indices:
			self.assertAlmostEqual(f.e[idx], 0.1)

		# set indices by vector
		d = np.random.rand(len(indices))
		f.set(range=indices, d=d)
		for i, idx in enumerate(indices):
			self.assertAlmostEqual(f.d[idx], d[i])

	def test_solver_object(self):
		s = PogsSolver(self.A_test)
		self.assertFalse( backend.device_reset_allowed )
		del s
		gc.collect()
		self.assertTrue( backend.device_reset_allowed )

	def test_solve_call(self):
		s = PogsSolver(self.A_test)
		f = PogsObjective(self.shape[0], h='Abs', b=1)
		g = PogsObjective(self.shape[1], h='IndGe0')
		s.solve(f, g)
		self.assertEqual(s.info.err, 0)
		self.assertTrue(s.info.converged or s.info.k == s.settings.maxiter)
		del s

	def test_solver_io(self):
		f = PogsObjective(self.shape[0], h='Abs', b=1)
		g = PogsObjective(self.shape[1], h='IndGe0')

		s = PogsSolver(self.A_test)

		cache_file = s.save(os.path.abspath('.'), 'c_solve_test')
		self.register_file(cache_file)

		s.solve(f, g, resume=0, maxiter=10000)

		s2 = PogsSolver(self.A_test, 'no_init')
		s2.load(os.path.abspath('.'), 'c_solve_test')
		s2.solve(f, g, resume=0, maxiter=10000)

		cache_file = s2.save(os.path.abspath('.'),'c_solve_test2')
		self.register_file(cache_file)

		s3 = PogsSolver(self.A_test, 'no_init')
		s3.load(os.path.abspath('.'), 'c_solve_test2')
		s3.solve(f, g, resume=1, maxiter=10000)

		factor = 30. if backend.pogs.pyfloat == np.float32 else 10.
		factor *= 2.**TEST_ITERATE
		self.assertTrue(s3.info.c.k <= s2.info.c.k or not s3.info.c.converged)

		diff_12 = abs(s2.info.c.obj - s.info.c.obj)
		diff_13 = abs(s3.info.c.obj - s.info.c.obj)

		tolerance = factor * s.settings.c.reltol * max(1, abs(s.info.objval))

		self.assertTrue(diff_12 <= tolerance or
						not s2.info.converged or
						not s.info.converged)

		self.assertTrue(diff_13 <= tolerance or
						not s3.info.converged or
						not s.info.converged)

		del s
		del s2
		del s3