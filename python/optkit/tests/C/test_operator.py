import unittest
import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_void_p, byref, CFUNCTYPE
from optkit.libs import DenseLinsysLibs, SparseLinsysLibs, OperatorLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
import pdb

class OperatorLibsTestCase(unittest.TestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.sparse_libs = SparseLinsysLibs()
		self.op_libs = OperatorLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
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

		self.x_test = np.random.rand(self.shape[1])
		self.nnz = sum(sum(self.A_test_sparse > 0))

	def validate_operator(self, operator_, m, n, OPERATOR_KIND):
		o = operator_
		self.assertEqual(o.size1, m)
		self.assertEqual(o.size2, n)
		self.assertEqual(o.kind, OPERATOR_KIND)
		self.assertNotEqual(o.data, 0)
		self.assertNotEqual(o.apply, 0)
		self.assertNotEqual(o.adjoint, 0)
		self.assertNotEqual(o.fused_apply, 0)
		self.assertNotEqual(o.fused_adjoint, 0)
		self.assertNotEqual(o.free, 0)

	def exercise_operator(self, dlib, operator_, A_py, TOL):
		o = operator_
		m, n = A_py.shape
		RTOL = TOL
		ATOLM = TOL * m**0.5
		ATOLN = TOL * n**0.5

		alpha = np.random.rand()
		beta = np.random.rand()

		# allocate vectors x, y
		x_ = np.zeros(n).astype(dlib.pyfloat)
		x_ += self.x_test
		x_ptr = x_.ctypes.data_as(dlib.ok_float_p)
		x = dlib.vector(0, 0, None)
		dlib.vector_calloc(x, n)
		dlib.vector_memcpy_va(x, x_ptr, 1)

		y_ = np.zeros(m).astype(dlib.pyfloat)
		y_ptr = y_.ctypes.data_as(dlib.ok_float_p)
		y = dlib.vector(0, 0, None)
		dlib.vector_calloc(y, m)

		# test Ax
		Ax = A_py.dot(x_)
		o.apply(o.data, x, y)
		dlib.vector_memcpy_av(y_ptr, y, 1)
		self.assertTrue(np.linalg.norm(y_ - Ax) <=
						ATOLM + RTOL * np.linalg.norm(Ax))

		# test A'y
		y_[:] = Ax[:] 	# (update for consistency)
		Aty = A_py.T.dot(y_)
		o.adjoint(o.data, y, x)
		dlib.vector_memcpy_av(x_ptr, x, 1)
		self.assertTrue(np.linalg.norm(x_ - Aty) <=
						ATOLN + RTOL * np.linalg.norm(Aty))

		# test Axpy
		x_[:]  = Aty[:] # (update for consistency)
		Axpy = alpha * A_py.dot(x_) + beta * y_
		o.fused_apply(o.data, alpha, x, beta, y)
		dlib.vector_memcpy_av(y_ptr, y, 1)
		self.assertTrue(np.linalg.norm(y_ - Axpy) <=
						ATOLM + RTOL * np.linalg.norm(Axpy))

		# test A'ypx
		y_[:] = Axpy[:] # (update for consistency)
		Atypx = alpha * A_py.T.dot(y_) + beta * x_
		o.fused_adjoint(o.data, alpha, y, beta, x)
		dlib.vector_memcpy_av(x_ptr, x, 1)
		self.assertTrue(np.linalg.norm(x_ - Atypx) <=
						ATOLN + RTOL * np.linalg.norm(Atypx))

		dlib.vector_free(x)
		dlib.vector_free(y)

	def test_libs_exist(self):
		dlibs = []
		slibs = []
		oplibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(gpu=gpu,
						 single_precision=single_precision))
			slibs.append(self.sparse_libs.get(
								dlibs[-1], single_precision=single_precision,
								gpu=gpu))
			oplibs.append(self.op_libs.get(
						  		dlibs[-1], slibs[-1],
						  		single_precision=single_precision, gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(oplibs))

	def test_dense_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			lib = self.op_libs.get(dlib, slib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor

				A = dlib.matrix(0, 0, 0, None, order)
				dlib.matrix_calloc(A, m, n, order)

				o = lib.dense_operator_alloc(A)
				self.validate_operator(o.contents, m, n, lib.enums.DENSE)
				lib.operator_free(o)

				dlib.matrix_free(A)
				self.assertEqual(dlib.ok_device_reset(), 0)

	def test_dense_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			lib = self.op_libs.get(dlib, slib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_ = np.zeros(self.shape, order=pyorder).astype(dlib.pyfloat)
				A_ += self.A_test
				A_ptr = A_.ctypes.data_as(dlib.ok_float_p)
				A = dlib.matrix(0, 0, 0, None, order)
				dlib.matrix_calloc(A, m, n, order)
				dlib.matrix_memcpy_ma(A, A_ptr, order)

				o = lib.dense_operator_alloc(A)
				self.exercise_operator(dlib, o.contents, A_, TOL)
				lib.operator_free(o)

				dlib.matrix_free(A)
				self.assertEqual(dlib.ok_device_reset(), 0)

	def test_sparse_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			lib = self.op_libs.get(dlib, slib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				spmat = csr_matrix if rowmajor else csc_matrix
				enum = lib.enums.SPARSE_CSR if rowmajor else \
					   lib.enums.SPARSE_CSC

				A = slib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
				slib.sp_matrix_calloc(A, m, n, self.nnz, order)

				o = lib.sparse_operator_alloc(A)
				self.validate_operator(o.contents, m, n, enum)
				lib.operator_free(o)

				slib.sp_matrix_free(A)
				self.assertEqual(dlib.ok_device_reset(), 0)

	def test_sparse_operator(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			lib = self.op_libs.get(dlib, slib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			TOL = 10**(-DIGITS)

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				spmat = csr_matrix if rowmajor else csc_matrix

				hdl = c_void_p()
				self.assertEqual(slib.sp_make_handle(byref(hdl)), 0)

				A_ = np.zeros(self.shape).astype(dlib.pyfloat)
				A_ += self.A_test_sparse
				A_sp = spmat(A_)
				A_val = A_sp.data.ctypes.data_as(dlib.ok_float_p)
				A_ind = A_sp.indices.ctypes.data_as(slib.ok_int_p)
				A_ptr = A_sp.indptr.ctypes.data_as(slib.ok_int_p)

				A = slib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
				slib.sp_matrix_calloc(A, m, n, A_sp.nnz, order)
				slib.sp_matrix_memcpy_ma(hdl, A, A_val, A_ind, A_ptr, order)

				o = lib.sparse_operator_alloc(A)
				self.exercise_operator(dlib, o.contents, A_, TOL)
				lib.operator_free(o)

				slib.sp_matrix_free(A)

				self.assertEqual(slib.sp_destroy_handle(hdl), 0)
				self.assertEqual(dlib.ok_device_reset(), 0)