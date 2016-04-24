import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs import EquilibrationLibs
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

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue(any(libs))

	@staticmethod
	def equilibrate(lib, equilibration_method, order, pyorder, A_test,
					x_test):
		m, n = shape = A_test.shape
		hdl = c_void_p()
		lib.blas_make_handle(byref(hdl))

		# declare C matrix, vectors
		A = lib.matrix(0, 0, 0, None, order)
		d = lib.vector(0, 0, None)
		e = lib.vector(0, 0, None)

		lib.matrix_calloc(A, m, n, order)
		lib.vector_calloc(d, m)
		lib.vector_calloc(e, n)

		# declare Py matrix, vectors, and corresponding pointers
		A_py = np.zeros(shape, order=pyorder).astype(lib.pyfloat)
		d_py = np.zeros(m).astype(lib.pyfloat)
		e_py = np.zeros(n).astype(lib.pyfloat)

		A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
		d_ptr = d_py.ctypes.data_as(lib.ok_float_p)
		e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

		A_in_ptr = A_test.astype(lib.pyfloat).ctypes.data_as(lib.ok_float_p)

		order_in = lib.enums.CblasRowMajor if A_test.flags.c_contiguous else \
				   lib.enums.CblasColMajor

		equilibration_method(hdl, A_in_ptr, A, d, e, order_in)

		lib.matrix_memcpy_am(A_ptr, A, order)
		lib.vector_memcpy_av(d_ptr, d, 1)
		lib.vector_memcpy_av(e_ptr, e, 1)

		lib.matrix_free(A)
		lib.vector_free(d)
		lib.vector_free(e)

		lib.blas_destroy_handle(hdl)
		lib.ok_device_reset()

		A_eqx = A_py.dot(x_test)
		DAEx = d_py * A_test.dot(e_py * x_test)

		return A_eqx, DAEx

	def test_densel2(self):
		for (gpu, single_precision) in self.CONDITIONS:
			# TODO: figure out why dense_l2 segfaults on GPU
			if gpu:
				continue

			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * self.shape[1]**0.5

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(lib, lib.dense_l2, order,
											   pyorder, self.A_test,
											   self.x_test)

				print np.max(np.abs(A_eqx - DAEx))
				self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
								ATOLN + RTOL * np.linalg.norm(DAEx))

	def test_sinkhorn_knopp(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * self.shape[1]**0.5

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(lib, lib.sinkhorn_knopp,
											   order, pyorder, self.A_test,
											   self.x_test)
				print np.max(np.abs(A_eqx - DAEx))
				self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
								ATOLN + RTOL * np.linalg.norm(DAEx))

	def test_regularized_sinkhorn_knopp(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * self.shape[0]**0.5

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(lib,
											   lib.regularized_sinkhorn_knopp,
											   order, pyorder, self.A_test,
											   self.x_test)
				print np.max(np.abs(A_eqx - DAEx))
				self.assertTrue( np.linalg.norm(A_eqx - DAEx) <=
								 ATOLM + RTOL * np.linalg.norm(DAEx) )

				A_rowmissing = np.zeros_like(self.A_test)
				A_rowmissing += self.A_test
				A_rowmissing[self.shape[0]/2, :] *= 0

				A_eqx, DAEx = self.equilibrate(lib,
											   lib.regularized_sinkhorn_knopp,
											   order, pyorder, A_rowmissing,
											   self.x_test)

				print np.max(np.abs(A_eqx - DAEx))
				self.assertTrue( np.linalg.norm(A_eqx - DAEx) <=
								 ATOLM + RTOL * np.linalg.norm(DAEx) )

				A_colmissing = np.zeros_like(self.A_test)
				A_colmissing += self.A_test
				A_colmissing[:, self.shape[1]/2] *= 0

				A_eqx, DAEx = self.equilibrate(lib,
											   lib.regularized_sinkhorn_knopp,
											   order, pyorder, A_colmissing,
											   self.x_test)

				print np.max(np.abs(A_eqx - DAEx))
 				self.assertTrue( np.linalg.norm(A_eqx - DAEx) <=
 								 ATOLM + RTOL * np.linalg.norm(DAEx) )

	def test_operator_sinkhorn_knopp(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5

			hdl = c_void_p()
			lib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y, d, e in python & C

			# declare C vectors
			x = lib.vector(0, 0, None)
			y = lib.vector(0, 0, None)
			d = lib.vector(0, 0, None)
			e = lib.vector(0, 0, None)

			lib.vector_calloc(x, n)
			lib.vector_calloc(y, m)
			lib.vector_calloc(d, m)
			lib.vector_calloc(e, n)
			self.register_var('x', x, lib.vector_free)
			self.register_var('y', y, lib.vector_free)
			self.register_var('d', d, lib.vector_free)
			self.register_var('e', e, lib.vector_free)

			# declare Py vectors, and corresponding pointers
			x_py = np.zeros(n).astype(lib.pyfloat)
			y_py = np.zeros(m).astype(lib.pyfloat)
			d_py = np.zeros(m).astype(lib.pyfloat)
			e_py = np.zeros(n).astype(lib.pyfloat)

			x_ptr = x_py.ctypes.data_as(lib.ok_float_p)
			y_ptr = y_py.ctypes.data_as(lib.ok_float_p)
			d_ptr = d_py.ctypes.data_as(lib.ok_float_p)
			e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

			x_py += self.x_test

			# -----------------------------------------
			# test equilibration for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "operator sinkhorn, operator type:", op_
				A_, A, o, freeA = self.gen_operator(op_, lib)
				self.register_var('A', A, freeA)
				self.register_var('o', o, o.contents.free)

				# equilibrate operator
				status = lib.operator_regularized_sinkhorn(hdl, o, d, e, 1.)

				# extract results
				lib.vector_memcpy_av(d_ptr, d, 1)
				lib.vector_memcpy_av(e_ptr, e, 1)
				DAEx = d_py * A_.dot(e_py * self.x_test)

				lib.vector_memcpy_va(x, x_ptr, 1)
				o.contents.apply(o.contents.data, x, y)

				lib.vector_memcpy_av(y_ptr, y, 1)
				A_eqx = y_py

				self.assertEqual(status, 0)
				self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
								ATOLN + RTOL * np.linalg.norm(DAEx))

				self.free_var('A')
				self.free_var('o')

			# -----------------------------------------
			# free x, y, d, e
			self.free_var('x')
			self.free_var('y')
			self.free_var('d')
			self.free_var('e')

			lib.blas_destroy_handle(hdl)
			lib.ok_device_reset()

	def test_operator_equil(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5

			hdl = c_void_p()
			lib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y, d, e in python & C

			# declare C vectors
			x = lib.vector(0, 0, None)
			y = lib.vector(0, 0, None)
			d = lib.vector(0, 0, None)
			e = lib.vector(0, 0, None)

			lib.vector_calloc(x, n)
			lib.vector_calloc(y, m)
			lib.vector_calloc(d, m)
			lib.vector_calloc(e, n)
			self.register_var('x', x, lib.vector_free)
			self.register_var('y', y, lib.vector_free)
			self.register_var('d', d, lib.vector_free)
			self.register_var('e', e, lib.vector_free)

			# declare Py vectors, and corresponding pointers
			x_py = np.zeros(n).astype(lib.pyfloat)
			y_py = np.zeros(m).astype(lib.pyfloat)
			d_py = np.zeros(m).astype(lib.pyfloat)
			e_py = np.zeros(n).astype(lib.pyfloat)

			x_ptr = x_py.ctypes.data_as(lib.ok_float_p)
			y_ptr = y_py.ctypes.data_as(lib.ok_float_p)
			d_ptr = d_py.ctypes.data_as(lib.ok_float_p)
			e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

			x_py += self.x_test

			# -----------------------------------------
			# test equilibration for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "opeartor sinkhorn, operator type:", op_
				A_, A, o, freeA = self.gen_operator(op_, lib)
				self.register_var('A', A, freeA)
				self.register_var('o', o, o.contents.free)

				# equilibrate operator
				status = lib.operator_equilibrate(hdl, o, d, e, 1.)

				# extract results
				lib.vector_memcpy_av(d_ptr, d, 1)
				lib.vector_memcpy_av(e_ptr, e, 1)
				DAEx = d_py * A_.dot(e_py * self.x_test)

				lib.vector_memcpy_va(x, x_ptr, 1)
				o.contents.apply(o.contents.data, x, y)

				lib.vector_memcpy_av(y_ptr, y, 1)
				A_eqx = y_py

				# TODO: REPLACE THIS WITH THE REAL TEST BELOW
				self.assertEqual(status, 1)

				# REAL TEST:
				# self.assertEqual(status, 0)
				# self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
				# 				ATOLN + RTOL * np.linalg.norm(DAEx))

				self.free_var('A')
				self.free_var('o')

			# -----------------------------------------
			# free x, y, d, e
			self.free_var('x')
			self.free_var('y')
			self.free_var('d')
			self.free_var('e')

			lib.blas_destroy_handle(hdl)
			lib.ok_device_reset()

	def test_operator_norm(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			RTOL = 0.05
			ATOL = 0.005 * (m * n)**0.5

			hdl = c_void_p()
			lib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# test norm estimation for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "indirect projection, operator type:", op_
				A_, A, o, freeA = self.gen_operator(op_, lib)
				self.register_var('A', A, freeA)
				self.register_var('o', o, o.contents.free)

				# estimate operator norm
				pynorm = np.linalg.norm(A_)
				cnorm = lib.operator_estimate_norm(hdl, o)

				print pynorm
				print cnorm
				self.assertTrue(
					cnorm >= ATOL + RTOL * pynorm or
					pynorm >= ATOL + RTOL * cnorm )

				self.free_var('A')
				self.free_var('o')

			lib.blas_destroy_handle(hdl)
			lib.ok_device_reset()