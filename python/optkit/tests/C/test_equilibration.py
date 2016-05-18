import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs.equilibration import EquilibrationLibs
from optkit.tests.C.base import OptkitCOperatorTestCase

class EquilLibsTestCase(OptkitCOperatorTestCase):
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
		self.libs = EquilibrationLibs()
		self.A_test = self.A_test_gen
		self.A_test_sparse = self.A_test_sparse_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue(any(libs))

	def equilibrate(self, lib, order, A_test):
		m, n = A_test.shape
		DIGITS = 7 - 2 * lib.FLOAT - 2 * lib.GPU
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5

		pyorder = 'C' if order == lib.enums.CblasRowMajor else 'F'
		hdl = self.register_blas_handle(lib, 'hdl')

		A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')
		d, d_py, d_ptr = self.register_vector(lib, m, 'd')
		e, e_py, e_ptr = self.register_vector(lib, n, 'e')

		A_in_py = A_test.astype(lib.pyfloat)
		A_in_ptr = A_in_py.ctypes.data_as(lib.ok_float_p)

		order_in = lib.enums.CblasRowMajor if A_in_py.flags.c_contiguous else \
				   lib.enums.CblasColMajor

		self.assertCall( lib.regularized_sinkhorn_knopp(
				hdl, A_in_ptr, A, d, e, order_in) )

		self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
		self.assertCall( lib.vector_memcpy_av(d_ptr, d, 1) )
		self.assertCall( lib.vector_memcpy_av(e_ptr, e, 1) )

		self.free_vars('A', 'd', 'e', 'hdl')
		self.assertCall( lib.ok_device_reset() )

		A_eqx = A_py.dot(self.x_test)
		DAEx = d_py * A_test.dot(e_py * self.x_test)

		self.assertVecEqual( A_eqx, DAEx, ATOLM, RTOL )

	def test_regularized_sinkhorn_knopp(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				print "regularized sinkhorn, CBLAS layout:", order

				self.equilibrate(lib, order, self.A_test)

				A_rowmissing = np.zeros_like(self.A_test)
				A_rowmissing += self.A_test
				A_rowmissing[self.shape[0]/2, :] *= 0

				self.equilibrate(lib, order, A_rowmissing)

				A_colmissing = np.zeros_like(self.A_test)
				A_colmissing += self.A_test
				A_colmissing[:, self.shape[1]/2] *= 0

				self.equilibrate(lib, order, A_colmissing)
				self.assertCall( lib.ok_device_reset() )

	def test_operator_sinkhorn_knopp(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5

			# -----------------------------------------
			# test equilibration for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "operator sinkhorn, operator type:", op_
				hdl = self.register_blas_handle(lib, 'hdl')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')
				d, d_py, d_ptr = self.register_vector(lib, m, 'd')
				e, e_py, e_ptr = self.register_vector(lib, n, 'e')
				x_py += self.x_test
				A_, A, o = self.register_operator(lib, op_)

				# equilibrate operator
				self.assertCall( lib.operator_regularized_sinkhorn(hdl, o, d,
																   e, 1.) )
				# extract results
				self.assertCall( lib.vector_memcpy_av(d_ptr, d, 1) )
				self.assertCall( lib.vector_memcpy_av(e_ptr, e, 1) )
				DAEx = d_py * A_.dot(e_py * self.x_test)

				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				self.assertCall( o.contents.apply(o.contents.data, x, y) )

				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				A_eqx = y_py

				self.assertVecEqual( A_eqx, DAEx, ATOLN, RTOL )

				self.free_vars('A', 'o', 'x', 'y', 'd', 'e', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_operator_equil(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5


			# -----------------------------------------
			# test equilibration for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "operator equil generic operator type:", op_
				hdl = self.register_blas_handle(lib, 'hdl')
				x, x_py, x_ptr = self.register_vector(lib, n, 'x')
				y, y_py, y_ptr = self.register_vector(lib, m, 'y')
				d, d_py, d_ptr = self.register_vector(lib, m, 'd')
				e, e_py, e_ptr = self.register_vector(lib, n, 'e')
				x_py += self.x_test

				A_, A, o = self.register_operator(lib, op_)

				# equilibrate operator
				status = lib.operator_equilibrate(hdl, o, d, e, 1.)

				# extract results
				self.assertCall( lib.vector_memcpy_av(d_ptr, d, 1) )
				self.assertCall( lib.vector_memcpy_av(e_ptr, e, 1) )
				DAEx = d_py * A_.dot(e_py * self.x_test)

				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
				o.contents.apply(o.contents.data, x, y)

				self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )
				A_eqx = y_py

				# TODO: REPLACE THIS WITH THE REAL TEST BELOW
				self.assertEqual(status, 1)

				# REAL TEST:
				# self.assertEqual( status, 0 )
				# self.assertVecEqual( A_eqx, DAEx, ATOLN, RTOL )
				self.free_vars('A', 'o', 'x', 'y', 'd', 'e', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_operator_norm(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			RTOL = 5e-2
			ATOL = 5e-3 * (m * n)**0.5

			# -----------------------------------------
			# test norm estimation for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "operator norm, operator type:", op_
				hdl = self.register_blas_handle(lib, 'hdl')
				A_, A, o = self.register_operator(lib, op_)

				# estimate operator norm
				normest_p = lib.ok_float_p()
				normest_p.contents = lib.ok_float(0.)

				pynorm = np.linalg.norm(A_)
				self.assertCall( lib.operator_estimate_norm(hdl, o,
					normest_p) )
				cnorm = normest_p.contents

				if self.VERBOSE_TEST:
					print "operator norm, Python: ", pynorm
					print "norm estimate, C: ", cnorm

				self.assertTrue(
					cnorm >= ATOL + RTOL * pynorm or
					pynorm >= ATOL + RTOL * cnorm )
				self.free_vars('A', 'o','hdl')
				self.assertCall( lib.ok_device_reset() )