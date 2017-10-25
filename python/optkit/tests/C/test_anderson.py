from optkit.compat import *

import os
import numpy as np
import numpy.linalg as la

from optkit.libs.anderson import AndersonLibs
from optkit.tests.C.base import OptkitCTestCase

class AndersonLibsTestCase(OptkitCTestCase):
	""" Python unit tests for optkit_anderson """

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = AndersonLibs()
		self.n = 5#0
		self.lookback = 3

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(
					single_precision=single_precision, gpu=gpu))
		self.assertTrue( any(libs) )

	def test_accelerator_alloc_free(self):
		n, lookback = self.n, self.lookback

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			x, x_py, x_ptr = self.register_vector(lib, n, 'x')

			aa = lib.anderson_accelerator_init(x, lookback)
			self.register_var('aa', aa, lib.anderson_accelerator_free)
			self.assertIsInstance( aa.contents.F, lib.matrix_p )
			self.assertIsInstance( aa.contents.G, lib.matrix_p )
			self.assertIsInstance( aa.contents.F_gram, lib.matrix_p )
			self.assertIsInstance( aa.contents.f, lib.vector_p )
			self.assertIsInstance( aa.contents.g, lib.vector_p )
			self.assertIsInstance( aa.contents.diag, lib.vector_p )
			self.assertIsInstance( aa.contents.alpha, lib.vector_p )
			self.assertIsInstance( aa.contents.ones, lib.vector_p )
			self.assertEqual( aa.contents.mu_regularization, 0.01 )
			self.assertEqual( aa.contents.iter, 0 )
			self.assertCall( lib.anderson_accelerator_free(aa) )
			self.unregister_var('aa')

			self.free_vars('x')
			self.assertCall( lib.ok_device_reset() )

	def test_accelerator_update_matrices(self):
		n, lookback = self.n, self.lookback

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				F, F_py, F_ptr = self.register_matrix(
						lib, n, lookback + 1, order, 'F')
				G, G_py, G_ptr = self.register_matrix(
						lib, n, lookback + 1, order, 'G')

				x, x_, x_ptr = self.register_vector(lib, n, 'x', random=True)
				g, g_, g_ptr = self.register_vector(lib, n, 'g', random=True)

				aa = lib.anderson_accelerator_init(x, lookback)
				self.register_var('aa', aa, lib.anderson_accelerator_free)

				i = int(lookback * np.random.rand(1))

				self.assertCall( lib.anderson_update_F_x(aa, F, x, i) )
				self.assertCall( lib.matrix_memcpy_am(F_ptr, F, order) )
				self.assertVecEqual( F_py[:, i], - x_, 1e-7, 1e-7 )

				self.assertCall( lib.anderson_update_F_g(aa, F, g, i) )
				self.assertCall( lib.matrix_memcpy_am(F_ptr, F, order) )
				self.assertVecEqual( F_py[:, i], g_ - x_, 1e-7, 1e-7 )

				self.assertCall( lib.anderson_update_G(aa, G, g, i) )
				self.assertCall( lib.matrix_memcpy_am(G_ptr, G, order) )
				self.assertVecEqual( G_py[:, i], g_, 1e-7, 1e-7 )

				self.free_vars('aa', 'F', 'G', 'g', 'x')
				self.assertCall( lib.ok_device_reset() )

	def test_accelerator_gramian(self):
		n, lookback = self.n, self.lookback

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)
			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOL = RTOL * (lookback + 1) # sqrt{lookback^2}

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				F, F_py, F_ptr = self.register_matrix(
						lib, n, lookback + 1, order, 'F', random=True)
				F_gram, F_gram_py, F_gram_ptr = self.register_matrix(
						lib, lookback + 1, lookback + 1, order, 'F_gram')
				x, x_, x_ptr = self.register_vector(lib, n, 'x')

				aa = lib.anderson_accelerator_init(x, lookback)
				self.register_var('aa', aa, lib.anderson_accelerator_free)

				F_gram_calc = F_py.T.dot(F_py)
				F_gram_calc += np.eye(lookback + 1) * np.sqrt(
						aa.contents.mu_regularization)

				self.assertCall( lib.anderson_regularized_gram(
						aa, F, F_gram, aa.contents.mu_regularization) )
				self.assertCall( lib.matrix_memcpy_am(
						F_gram_ptr, F_gram, order) )
				self.assertVecEqual( F_gram_py, F_gram_calc, ATOL, RTOL)

				self.free_vars('aa', 'F', 'F_gram', 'x')
				self.assertCall( lib.ok_device_reset() )

	@staticmethod
	def py_anderson_solve(F, mu):
		m = F.shape[1]
		F_gram = F.T.dot(F) + np.sqrt(mu) * np.eye(m)
		alpha = la.solve(F_gram, np.ones(m))
		return alpha / np.sum(alpha)

	def test_anderson_solve(self):
		n, lookback = self.n, self.lookback

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)
			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOL = RTOL * (lookback + 1)**0.5
			ORDER = lib.enums.CblasColMajor

			F, F_py, F_ptr = self.register_matrix(
					lib, n, lookback + 1, ORDER, 'F', random=True)
			x, x_, x_ptr = self.register_vector(lib, n, 'x')
			alpha, alpha_py, alpha_ptr = self.register_vector(
					lib, lookback + 1, 'alpha')

			aa = lib.anderson_accelerator_init(x, lookback)
			self.register_var('aa', aa, lib.anderson_accelerator_free)

			####
			alpha__ = self.py_anderson_solve(
					F_py, aa.contents.mu_regularization)
			####

			self.assertCall( lib.anderson_solve(
					aa, F, alpha, aa.contents.mu_regularization) )
			self.assertCall( lib.vector_memcpy_av(alpha_ptr, alpha, 1) )
			self.assertVecEqual( alpha_py, alpha__, ATOL, RTOL )

			self.free_vars('aa', 'F', 'x', 'alpha')
			self.assertCall( lib.ok_device_reset() )

	def test_anderson_mix(self):
		n, lookback = self.n, self.lookback

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)
			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOL = RTOL * n**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				G, G_py, G_ptr = self.register_matrix(
						lib, n, lookback + 1, order, 'G', random=True)

				alpha, alpha_py, alpha_ptr = self.register_vector(
						lib, lookback + 1, 'alpha', random=True)
				x, x_py, x_ptr = self.register_vector(lib, n, 'x', random=True)

				aa = lib.anderson_accelerator_init(x, lookback)
				self.register_var('aa', aa, lib.anderson_accelerator_free)
				self.assertCall( lib.anderson_mix(aa, G, alpha, x) )

				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( x_py, np.dot(G_py, alpha_py), ATOL, RTOL )

				self.free_vars('aa', 'G', 'x', 'alpha')
				self.assertCall( lib.ok_device_reset() )

	def test_anderson_accelerate(self):
		n, lookback = self.n, self.lookback

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)
			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOL = RTOL * n**0.5

			x, x_py, x_ptr = self.register_vector(lib, n, 'x')

			F = np.zeros((n, lookback + 1))
			G = np.zeros((n, lookback + 1))

			#TODO: TEST THINGS HERE
			aa = lib.anderson_accelerator_init(x, lookback)
			mu = aa.contents.mu_regularization

			# TEST THE BEHAVIOR < LOOKBACK AND >= LOOKBACK
			for k in xrange(lookback + 5):
				index = k % (lookback + 1)
				index_next = (k + 1) % (lookback + 1)

				xcurr = np.random.rand(n)
				x_py[:] = xcurr

				F[:, index] += xcurr
				G[:, index] = xcurr

				if k < lookback:
					xnext = xcurr
				else:
					xnext = np.dot(G, self.py_anderson_solve(F, mu))

				F[:, index_next] = -xnext

				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( lib.anderson_accelerate(aa, x) )

				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( xnext, x_py, ATOL, RTOL )

			self.free_vars('aa', 'x')
			self.assertCall( lib.ok_device_reset() )