import unittest
import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs import DenseLinsysLibs, EquilibrationLibs
from optkit.tests.defs import CONDITIONS

class EquilLibsTestCase(unittest.TestCase):
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
		self.equil_libs = EquilibrationLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = (150, 250)
		self.A_test = np.random.rand(*self.shape)
		self.x_test = np.random.rand(self.shape[1])

	def test_libs_exist(self):
		dlibs = []
		eqlibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(gpu=gpu,
						 single_precision=single_precision))
			eqlibs.append(self.equil_libs.get(
						  		dlibs[-1], single_precision=single_precision,
						  		gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(eqlibs))

	@staticmethod
	def equilibrate(dlib, equilibration_method, hdl, pytype, order, pyorder,
					shape, A_test, x_test):

		m, n = shape

		# declare C matrix, vectors
		A = dlib.matrix(0, 0, 0, None, order)
		d = dlib.vector(0, 0, None)
		e = dlib.vector(0, 0, None)

		dlib.matrix_calloc(A, m, n, order)
		dlib.vector_calloc(d, m)
		dlib.vector_calloc(e, n)

		# declare Py matrix, vectors, and corresponding pointers
		A_py = np.zeros(shape, order=pyorder).astype(pytype)
		d_py = np.zeros(m).astype(pytype)
		e_py = np.zeros(n).astype(pytype)

		A_ptr = A_py.ctypes.data_as(dlib.ok_float_p)
		d_ptr = d_py.ctypes.data_as(dlib.ok_float_p)
		e_ptr = e_py.ctypes.data_as(dlib.ok_float_p)

		A_in_ptr = A_test.astype(pytype).ctypes.data_as(dlib.ok_float_p)

		order_in = dlib.enums.CblasRowMajor if A_test.flags.c_contiguous else \
				   dlib.enums.CblasColMajor

		equilibration_method(hdl, A_in_ptr, A, d, e, order_in)
		dlib.matrix_memcpy_am(A_ptr, A, order)
		dlib.vector_memcpy_av(d_ptr, d, 1)
		dlib.vector_memcpy_av(e_ptr, e, 1)

		dlib.matrix_free(A)
		dlib.vector_free(d)
		dlib.vector_free(e)

		A_eqx = A_py.dot(x_test)
		DAEx = d_py * A_test.dot(e_py * x_test)

		return A_eqx, DAEx


	def test_densel2(self):
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.equil_libs.get(dlib, single_precision=single_precision,
									  gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)
			pytype = np.float32 if single_precision else np.float64

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(dlib, lib.dense_l2, hdl,
											   pytype, order, pyorder,
											   self.shape, self.A_test,
											   self.x_test)

				self.assertTrue(np.allclose(A_eqx, DAEx))

			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_sinkhorn_knopp(self):
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.equil_libs.get(dlib, single_precision=single_precision,
									  gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)
			pytype = np.float32 if single_precision else np.float64

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(dlib, lib.sinkhorn_knopp, hdl,
											   pytype, order, pyorder,
											   self.shape, self.A_test,
											   self.x_test)

				self.assertTrue(np.allclose(A_eqx, DAEx))

			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	# TODO:
	# test regularized sinkhorn-knopp (once it exists)