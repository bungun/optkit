import unittest
import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs import DenseLinsysLibs, EquilibrationLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH

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
	def equilibrate(dlib, equilibration_method, order, pyorder, A_test,
					x_test):
		m, n = shape = A_test.shape
		hdl = c_void_p()
		dlib.blas_make_handle(byref(hdl))

		# declare C matrix, vectors
		A = dlib.matrix(0, 0, 0, None, order)
		d = dlib.vector(0, 0, None)
		e = dlib.vector(0, 0, None)

		dlib.matrix_calloc(A, m, n, order)
		dlib.vector_calloc(d, m)
		dlib.vector_calloc(e, n)

		# declare Py matrix, vectors, and corresponding pointers
		A_py = np.zeros(shape, order=pyorder).astype(dlib.pyfloat)
		d_py = np.zeros(m).astype(dlib.pyfloat)
		e_py = np.zeros(n).astype(dlib.pyfloat)

		A_ptr = A_py.ctypes.data_as(dlib.ok_float_p)
		d_ptr = d_py.ctypes.data_as(dlib.ok_float_p)
		e_ptr = e_py.ctypes.data_as(dlib.ok_float_p)

		A_in_ptr = A_test.astype(dlib.pyfloat).ctypes.data_as(dlib.ok_float_p)

		order_in = dlib.enums.CblasRowMajor if A_test.flags.c_contiguous else \
				   dlib.enums.CblasColMajor

		equilibration_method(hdl, A_in_ptr, A, d, e, order_in)

		dlib.matrix_memcpy_am(A_ptr, A, order)
		dlib.vector_memcpy_av(d_ptr, d, 1)
		dlib.vector_memcpy_av(e_ptr, e, 1)

		dlib.matrix_free(A)
		dlib.vector_free(d)
		dlib.vector_free(e)

		dlib.blas_destroy_handle(hdl)
		dlib.ok_device_reset()

		A_eqx = A_py.dot(x_test)
		DAEx = d_py * A_test.dot(e_py * x_test)

		return A_eqx, DAEx

	def test_densel2(self):

		for (gpu, single_precision) in CONDITIONS:
			# TODO: figure out why dense_l2 segfaults on GPU
			if gpu:
				continue

			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.equil_libs.get(dlib, single_precision=single_precision,
									  gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(dlib, lib.dense_l2, order,
											   pyorder, self.A_test,
											   self.x_test)

				print np.max(A_eqx - DAEx)
				self.assertTrue(np.max(A_eqx - DAEx) <= 10**(-DIGITS))

	def test_sinkhorn_knopp(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.equil_libs.get(dlib, single_precision=single_precision,
									  gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(dlib, lib.sinkhorn_knopp,
											   order, pyorder, self.A_test,
											   self.x_test)
				print np.max(A_eqx - DAEx)
				self.assertTrue(np.max(A_eqx - DAEx) <= 10**(-DIGITS))

	def test_regularized_sinkhorn_knopp(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			lib = self.equil_libs.get(dlib, single_precision=single_precision,
									  gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(dlib,
											   lib.regularized_sinkhorn_knopp,
											   order, pyorder, self.A_test,
											   self.x_test)
				print np.max(A_eqx - DAEx)
				self.assertTrue(np.max(A_eqx - DAEx) <= 10**(-DIGITS))

				A_rowmissing = np.zeros_like(self.A_test)
				A_rowmissing += self.A_test
				A_rowmissing[self.shape[0]/2, :] *= 0

				A_eqx, DAEx = self.equilibrate(dlib,
											   lib.regularized_sinkhorn_knopp,
											   order, pyorder, A_rowmissing,
											   self.x_test)

				print np.max(A_eqx - DAEx)
				self.assertTrue(np.max(A_eqx - DAEx) <= 10**(-DIGITS))

				A_colmissing = np.zeros_like(self.A_test)
				A_colmissing += self.A_test
				A_colmissing[:, self.shape[1]/2] *= 0

				A_eqx, DAEx = self.equilibrate(dlib,
											   lib.regularized_sinkhorn_knopp,
											   order, pyorder, A_colmissing,
											   self.x_test)

				print np.max(A_eqx - DAEx)
				self.assertTrue(np.max(A_eqx - DAEx) <= 10**(-DIGITS))
