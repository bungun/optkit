import unittest
import os
import numpy as np
from ctypes import c_void_p, c_size_t, byref
from optkit.libs import DenseLinsysLibs
from optkit.libs.clustering import ClusteringLibs
from optkit.libs.error import optkit_print_error as PRINTERR
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
from optkit.tests.C.base import OptkitCTestCase

class ClusterLibsTestCase(OptkitCTestCase):
	"""
		Equilibrate input A_in as

			D * A_equil * E

		with D, E, diagonal.

		Test that

			D * A_equil * E == A_in,

		or that

			D^-1 * A_in * E^-1 == A_equil.
	"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.cluster_libs = ClusteringLibs()

		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)

		self.x_test = np.random.rand(self.shape[1])
		self.k = self.shape[0] / 3
		self.u_test = (self.k * np.random.rand(self.shape[0])).astype(c_size_t)

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()

	def test_libs_exist(self):
		dlibs = []
		cluslibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(
					gpu=gpu, single_precision=single_precision))
			cluslibs.append(self.cluster_libs.get(
					dlibs[-1], gpu=gpu, single_precision=single_precision))

		self.assertTrue(any(dlibs))
		self.assertTrue(any(cluslibs))

	def test_upsamplingvec_alloc_free(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			u = lib.upsamplingvec(0, 0, 0, None, None)
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)

			self.assertEqual( err, 0 )
			self.assertEqual( u.size1, m )
			self.assertEqual( u.size2, k )
			err = PRINTERR( lib.upsamplingvec_free(u) )
			self.assertEqual( err, 0 )

	def test_upsamplingvec_check_bounds(self):
		m, n = self.shape
		k = self.k

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.cluster_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			u_py = np.zeros(m).astype(c_size_t)
			u_ptr = u_py.ctypes.data_as(dlib.c_size_t_p)

			u = lib.upsamplingvec(0, 0, 0, None, None)
			err = PRINTERR( lib.upsamplingvec_alloc(u, m, k) )
			self.register_var('u', u, lib.upsamplingvec_free)

			err = PRINTERR( lib.upsamplingvec_check_bounds(u) )
			self.assertEqual( err, 0 )
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)
			err = PRINTERR( lib.upsamplingvec_check_bounds(u) )
			self.assertEqual( err, 0 )
			u_py += 2 * k
			dlib.indvector_memcpy_va(u.vec, u_ptr, 1)
			print "\nexpecting error:"
			err = PRINTERR( lib.upsamplingvec_check_bounds(u) )
			self.assertNotEqual( err, 0 )

			err = PRINTERR( lib.upsamplingvec_free(u) )

	def test_upsamplingvec_subvector(self):
		pass

	def test_upsamplingvec_mul_matrix(self):
		pass

	def test_upsamplingvec_count(self):
		pass

	def test_upsamplingvec_shift(self):
		pass

