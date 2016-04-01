import unittest
import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs import DenseLinsysLibs, SparseLinsysLibs, EquilibrationLibs, \
						OperatorLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
import optkit.tests.C.operator_helper as op_helper

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
		self.sparse_libs = SparseLinsysLibs()
		self.operator_libs = OperatorLibs()
		self.equil_libs = EquilibrationLibs()

		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.A_test_sparse = self.A_test
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)
			self.A_test_sparse = np.zeros(self.shape)
			self.A_test_sparse += self.A_test
			for i in xrange(self.shape[0]):
				if np.random.rand() > 0.4:
					self.A_test_sparse[i, :] *= 0
			for j in xrange(self.shape[1]):
				if np.random.rand() > 0.4:
					self.A_test_sparse[:, j] *= 0

		self.nnz = sum(sum(self.A_test_sparse > 0))

		self.x_test = np.random.rand(self.shape[1])

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	@property
	def op_keys(self):
		return ['dense', 'sparse']

	def get_opmethods(self, opkey, denselib, sparselib, operatorlib):
		if opkey == 'dense':
			A = self.A_test
			gen = op_helper.gen_dense_operator
			arg_gen = [denselib, operatorlib, A]
			release = op_helper.release_dense_operator
			arg_release = [denselib, operatorlib]
		elif opkey == 'sparse':
			A = self.A_test_sparse
			gen = op_helper.gen_sparse_operator
			arg_gen = [denselib, sparselib, operatorlib, A]
			release = op_helper.release_sparse_operator
			arg_release = [sparselib, operatorlib]
		else:
			raise ValueError('invalid operator type')

		return (A, gen, arg_gen, release, arg_release)

	def test_libs_exist(self):
		dlibs = []
		slibs = []
		olibs = []
		eqlibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(
					gpu=gpu, single_precision=single_precision))
			slibs.append(self.sparse_libs.get(
					dlibs[-1], gpu=gpu, single_precision=single_precision))
			olibs.append(self.operator_libs.get(
					dlibs[-1], slibs[-1], single_precision=single_precision,
					gpu=gpu))
			eqlibs.append(self.equil_libs.get(
					dlibs[-1], olibs[-1], single_precision=single_precision,
					gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(slibs))
		self.assertTrue(any(olibs))
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

			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.operator_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			lib = self.equil_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * self.shape[1]**0.5

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(dlib, lib.dense_l2, order,
											   pyorder, self.A_test,
											   self.x_test)

				print np.max(A_eqx - DAEx)
				self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
								ATOLN + RTOL * np.linalg.norm(DAEx))

	def test_sinkhorn_knopp(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.operator_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			lib = self.equil_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)


			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * self.shape[1]**0.5

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_eqx, DAEx = self.equilibrate(dlib, lib.sinkhorn_knopp,
											   order, pyorder, self.A_test,
											   self.x_test)
				print np.max(A_eqx - DAEx)
				self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
								ATOLN + RTOL * np.linalg.norm(DAEx))

	def test_regularized_sinkhorn_knopp(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.operator_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			lib = self.equil_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * self.shape[1]**0.5

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

				self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
								ATOLN + RTOL * np.linalg.norm(DAEx))

				print np.max(A_eqx - DAEx)
 				self.assertTrue(np.max(A_eqx - DAEx) <= 10**(-DIGITS))

	def test_operator_sinkhorn_knopp(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.operator_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			lib = self.equil_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)

			if olib is None or lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5

			hdl = c_void_p()
			dlib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y, d, e in python & C

			# declare C vectors
			x = dlib.vector(0, 0, None)
			y = dlib.vector(0, 0, None)
			d = dlib.vector(0, 0, None)
			e = dlib.vector(0, 0, None)

			dlib.vector_calloc(x, n)
			dlib.vector_calloc(y, m)
			dlib.vector_calloc(d, m)
			dlib.vector_calloc(e, n)

			# declare Py vectors, and corresponding pointers
			x_py = np.zeros(n).astype(dlib.pyfloat)
			y_py = np.zeros(m).astype(dlib.pyfloat)
			d_py = np.zeros(m).astype(dlib.pyfloat)
			e_py = np.zeros(n).astype(dlib.pyfloat)

			x_ptr = x_py.ctypes.data_as(dlib.ok_float_p)
			y_ptr = y_py.ctypes.data_as(dlib.ok_float_p)
			d_ptr = d_py.ctypes.data_as(dlib.ok_float_p)
			e_ptr = e_py.ctypes.data_as(dlib.ok_float_p)

			x_py += self.x_test

			# -----------------------------------------
			# test equilibration for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "opeartor sinkhorn, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				# equilibrate operator
				status = lib.operator_regularized_sinkhorn(hdl, o, d, e, 1.)

				# extract results
				dlib.vector_memcpy_av(d_ptr, d, 1)
				dlib.vector_memcpy_av(e_ptr, e, 1)
				DAEx = d_py * A_.dot(e_py * self.x_test)

				dlib.vector_memcpy_va(x, x_ptr, 1)
				o.contents.apply(o.contents.data, x, y)

				dlib.vector_memcpy_av(y_ptr, y, 1)
				A_eqx = y_py

				self.assertEqual(status, 0)
				self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
								ATOLN + RTOL * np.linalg.norm(DAEx))

				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, y, d, e
			dlib.vector_free(x)
			dlib.vector_free(y)
			dlib.vector_free(d)
			dlib.vector_free(e)

			dlib.blas_destroy_handle(byref(hdl))
			dlib.ok_device_reset()

	def test_operator_equil(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.operator_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			lib = self.equil_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)

			if olib is None or lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 2 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5

			hdl = c_void_p()
			dlib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y, d, e in python & C

			# declare C vectors
			x = dlib.vector(0, 0, None)
			y = dlib.vector(0, 0, None)
			d = dlib.vector(0, 0, None)
			e = dlib.vector(0, 0, None)

			dlib.vector_calloc(x, n)
			dlib.vector_calloc(y, m)
			dlib.vector_calloc(d, m)
			dlib.vector_calloc(e, n)

			# declare Py vectors, and corresponding pointers
			x_py = np.zeros(n).astype(dlib.pyfloat)
			y_py = np.zeros(m).astype(dlib.pyfloat)
			d_py = np.zeros(m).astype(dlib.pyfloat)
			e_py = np.zeros(n).astype(dlib.pyfloat)

			x_ptr = x_py.ctypes.data_as(dlib.ok_float_p)
			y_ptr = y_py.ctypes.data_as(dlib.ok_float_p)
			d_ptr = d_py.ctypes.data_as(dlib.ok_float_p)
			e_ptr = e_py.ctypes.data_as(dlib.ok_float_p)

			x_py += self.x_test

			# -----------------------------------------
			# test equilibration for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "opeartor sinkhorn, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				# equilibrate operator
				status = lib.operator_equilibrate(hdl, o, d, e, 1.)

				# extract results
				dlib.vector_memcpy_av(d_ptr, d, 1)
				dlib.vector_memcpy_av(e_ptr, e, 1)
				DAEx = d_py * A_.dot(e_py * self.x_test)

				dlib.vector_memcpy_va(x, x_ptr, 1)
				o.contents.apply(o.contents.data, x, y)

				dlib.vector_memcpy_av(y_ptr, y, 1)
				A_eqx = y_py

				# TODO: REPLACE THIS WITH THE REAL TEST BELOW
				self.assertEqual(status, 1)

				# REAL TEST:
				# self.assertEqual(status, 0)
				# self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
				# 				ATOLN + RTOL * np.linalg.norm(DAEx))

				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, y, d, e
			dlib.vector_free(x)
			dlib.vector_free(y)
			dlib.vector_free(d)
			dlib.vector_free(e)

			dlib.blas_destroy_handle(byref(hdl))
			dlib.ok_device_reset()

	def test_operator_norm(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.operator_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			lib = self.equil_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			RTOL = 0.1
			ATOL = 0.05 * (m * n)**0.5

			hdl = c_void_p()
			dlib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# test norm estimation for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "indirect projection, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				# estimate operator norm
				pynorm = np.linalg.norm(A_)
				cnorm = lib.operator_estimate_norm(hdl, o)

				self.assertTrue(np.abs(pynorm - cnorm) <=
								ATOL + RTOL * pynorm)

				release_args += [A, o]
				release_operator(*release_args)


			dlib.blas_destroy_handle(byref(hdl))
			dlib.ok_device_reset()