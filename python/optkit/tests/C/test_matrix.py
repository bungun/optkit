import unittest
import os
import numpy as np
from optkit.libs import DenseLinsysLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH

class MatrixTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()

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

	@staticmethod
	def make_mat_triplet(lib, shape, rowmajor=True):
		order = 101 if rowmajor else 102
		pyorder = 'C' if rowmajor else 'F'
		A = lib.matrix(0, 0, 0, None, order)
		lib.matrix_calloc(A, shape[0], shape[1], order)
		A_py = np.zeros(shape, order=pyorder).astype(lib.pyfloat)
		A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
		return A, A_py, A_ptr

	def test_alloc(self):
		(m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				A = lib.matrix(0, 0, 0, None, order)
				self.assertEqual(A.size1, 0)
				self.assertEqual(A.size2, 0)
				self.assertEqual(A.ld, 0)
				self.assertEqual(A.order, order)

				# calloc
				lib.matrix_calloc(A, m, n, order)
				self.assertEqual(A.size1, m)
				self.assertEqual(A.size2, n)
				if rowmajor:
					self.assertEqual(A.ld, n)
				else:
					self.assertEqual(A.ld, m)
				self.assertEqual(A.order, order)
				if not gpu:
					for i in xrange(m * n):
						self.assertEqual(A.data[i], 0)
				# free
				lib.matrix_free(A)

			self.assertEqual(lib.ok_device_reset(), 0)

	def test_io(self):
		(m, n) = self.shape
		A_rand = self.A_test
		x_rand = np.random.rand(n)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n), rowmajor)

				# memcpy_am
				# set A_py to A_rand. overwrite A_py with zeros from A
				A_py += A_rand
				lib.matrix_memcpy_am(A_ptr, A, order)
				for i in xrange(m):
					for j in xrange(n):
						self.assertEqual(A_py[i, j], 0)

				# memcpy_ma
				A_py += A_rand
				lib.matrix_memcpy_ma(A, A_ptr, order)
				A_py *= 0
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				# memcpy_mm
				Z, Z_py, Z_ptr = self.make_mat_triplet(lib, (m, n), rowmajor)
				lib.matrix_memcpy_mm(Z, A, order)
				lib.matrix_memcpy_am(Z_ptr, Z, order)
				self.assertTrue(np.allclose(Z_py, A_py, DIGITS))

				# view_array
				if not gpu:
					A_py *= 0
					B = lib.matrix(0, 0, 0, None, order)
					lib.matrix_view_array(B,
						A_rand.ctypes.data_as(lib.ok_float_p), m, n, order)
					lib.matrix_memcpy_am(A_ptr, A, order)
					self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				# set_all
				val = 2
				A_rand *= 0
				A_rand += val
				lib.matrix_set_all(A, val)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				lib.matrix_free(A)
				lib.matrix_free(Z)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_slicing(self):
		(m, n) = self.shape
		A_rand = self.A_test
		x_rand = np.random.rand(n)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n),
								   					   rowmajor)

				# set A, A_py to A_rand
				A_py += A_rand
				lib.matrix_memcpy_ma(A, A_ptr, order)

				# submatrix
				m0 = m / 4
				n0 = n / 4
				msub = m / 2
				nsub = n / 2
				Asub = lib.matrix(0, 0, 0, None, order)
				Asub_py = np.zeros(
						(msub, nsub), order=pyorder).astype(lib.pyfloat)
				Asub_ptr = Asub_py.ctypes.data_as(lib.ok_float_p)

				lib.matrix_submatrix(Asub, A, m0, n0, msub, nsub)
				lib.matrix_memcpy_am(Asub_ptr, Asub, order)
				self.assertTrue(np.allclose(A_py[m0 : m0+msub, n0 : n0+nsub],
											Asub_py, DIGITS))

				# row
				v = lib.vector(0, 0, None)
				v_py = np.zeros(n).astype(lib.pyfloat)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_row(v, A, m0)
				lib.vector_memcpy_av(v_ptr, v, 1)
				self.assertTrue(np.allclose(A_py[m0, :], v_py, DIGITS))

				# column
				v_py = np.zeros(m).astype(lib.pyfloat)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_column(v, A, n0)
				lib.vector_memcpy_av(v_ptr, v, 1)
				self.assertTrue(np.allclose(A_py[: , n0], v_py, DIGITS))

				# diagonal
				v_py = np.zeros(min(m, n)).astype(lib.pyfloat)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_diagonal(v, A)
				lib.vector_memcpy_av(v_ptr, v, 1)
				self.assertTrue(np.allclose(np.diag(A_py), v_py, DIGITS))

				lib.matrix_free(A)
			self.assertEqual(lib.ok_device_reset(), 0)


	def test_math(self):
		(m, n) = self.shape
		A_rand = self.A_test

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n), rowmajor)

				# set A, A_py to A_rand
				A_py += A_rand
				lib.matrix_memcpy_ma(A, A_ptr, order)

				# scale: A = alpha * A
				alpha = np.random.rand()
				A_rand *= alpha
				lib.matrix_scale(A, alpha)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				# scale_left: A = diag(d) * A
				d = lib.vector(0, 0, None)
				lib.vector_calloc(d, m)
				d_py = np.zeros(m).astype(lib.pyfloat)
				d_ptr = d_py.ctypes.data_as(lib.ok_float_p)
				d_py[:] = np.random.rand(m)
				for i in xrange(m):
					A_rand[i, :] *= d_py[i]
				lib.vector_memcpy_va(d, d_ptr, 1)
				lib.matrix_scale_left(A, d)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				# scale_right: A = A * diag(e)
				e = lib.vector(0, 0, None)
				lib.vector_calloc(e, n)
				e_py = np.zeros(n).astype(lib.pyfloat)
				e_ptr = e_py.ctypes.data_as(lib.ok_float_p)
				e_py[:] = np.random.rand(n)
				for j in xrange(n):
					A_rand[:, j] *= e_py[j]
				lib.vector_memcpy_va(e, e_ptr, 1)
				lib.matrix_scale_right(A, e)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				# abs: A_ij = abs(A_ij)
				A_rand -= (A_rand.max() - A_rand.min()) / 2
				A_py *= 0
				A_py += A_rand
				A_rand = np.abs(A_rand)
				lib.matrix_memcpy_ma(A, A_ptr, order)
				lib.matrix_abs(A)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				# pow
				p = 3 * np.random.rand()
				A_rand **= p
				lib.matrix_pow(A, p)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand, DIGITS))

				lib.vector_free(d)
				lib.vector_free(e)
				lib.matrix_free(A)

			self.assertEqual(lib.ok_device_reset(), 0)

	def test_diag_gramian(self):
		(m, n) = self.shape
		mindim = min(m, n)

		# Python: calculate diag of (AA') (A fat) or (A'A) (A skinny)
		Acols = self.A_test if m >= n else self.A_test.T
		py_diag = np.zeros(mindim)
		for j in xrange(mindim):
			py_diag[j] = Acols[:, j].dot(Acols[:, j])

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
				single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLMIN = RTOL * mindim**0.5

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				# allocate A, x
				A = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(A, m, n, order)
				A_py = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A_py += self.A_test
				A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_memcpy_ma(A, A_ptr, order)

				x = lib.vector(0, 0, None)
				lib.vector_calloc(x, mindim, order)
				x_py = np.zeros(mindim).astype(lib.pyfloat)
				x_ptr = x_py.ctypes.data_as(lib.ok_float_p)

				# C: calculate diag of (AA') (A fat) or (A'A) (A skinny)
				lib.linalg_diag_gramian(A, x)
				lib.vector_memcpy_av(x_ptr, x, 1)

				# compare C vs Python results
				self.assertTrue(np.linalg.norm(x_py - py_diag) <=
								ATOLMIN + RTOL * np.linalg.norm(py_diag))

				# free memory
				lib.matrix_free(A)
				lib.vector_free(x)

			self.assertEqual(lib.ok_device_reset(), 0)

	def test_broadcast(self):
		(m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
				single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				# allocate A, d, e
				A = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(A, m, n, order)
				A_py = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A_py += self.A_test
				A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_memcpy_ma(A, A_ptr, order)

				d = lib.vector(0, 0, None)
				lib.vector_calloc(d, m)
				d_py = np.zeros(m).astype(lib.pyfloat)
				d_ptr = d_py.ctypes.data_as(lib.ok_float_p)

				e = lib.vector(0, 0, None)
				lib.vector_calloc(e, n)
				e_py = np.zeros(n).astype(lib.pyfloat)
				e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

				x = lib.vector(0, 0, None)
				lib.vector_calloc(x, n)
				x_py = np.zeros(n).astype(lib.pyfloat)
				x_ptr = x_py.ctypes.data_as(lib.ok_float_p)

				y = lib.vector(0, 0, None)
				lib.vector_calloc(y, m)
				y_py = np.zeros(m).astype(lib.pyfloat)
				y_ptr = y_py.ctypes.data_as(lib.ok_float_p)

				d_py += np.random.rand(m)
				e_py += np.random.rand(n)
				x_py += np.random.rand(n)
				lib.vector_memcpy_va(d, d_ptr, 1)
				lib.vector_memcpy_va(e, e_ptr, 1)
				lib.vector_memcpy_va(x, x_ptr, 1)

				# A = A * diag(E)
				lib.linalg_matrix_broadcast_vector(hdl, A, e,
					lib.enums.OkTransformScale, lib.enums.CblasRight)

				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				AEx = self.A_test.dot(e_py * x_py)
				self.assertTrue(np.linalg.norm(Ax - AEx) <=
								ATOLM + RTOL * np.linalg.norm(AEx))

				# A = diag(D) * A
				lib.linalg_matrix_broadcast_vector(hdl, A, d,
					lib.enums.OkTransformScale, lib.enums.CblasLeft)
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				DAEx = d_py * AEx
				self.assertTrue(np.linalg.norm(Ax - DAEx) <=
								ATOLM + RTOL * np.linalg.norm(DAEx))

				# A += 1e'
				lib.linalg_matrix_broadcast_vector(hdl, A, e,
					lib.enums.OkTransformAdd, lib.enums.CblasRight)
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				A_updatex = DAEx + np.ones(m) * e_py.dot(x_py)
				self.assertTrue(np.linalg.norm(Ax - A_updatex) <=
								ATOLM + RTOL * np.linalg.norm(A_updatex))

				# A += d1'
				lib.linalg_matrix_broadcast_vector(hdl, A, d,
					lib.enums.OkTransformAdd, lib.enums.CblasLeft)
				lib.blas_gemv(hdl, lib.enums.CblasNoTrans, 1, A, x, 0, y)
				lib.vector_memcpy_av(y_ptr, y, 1)
				Ax = y_py
				A_updatex += d_py * sum(x_py)
				self.assertTrue(np.linalg.norm(Ax - A_updatex) <=
								ATOLM + RTOL * np.linalg.norm(A_updatex))

				# free memory
				lib.matrix_free(A)
				lib.vector_free(d)
				lib.vector_free(e)
				lib.vector_free(x)
				lib.vector_free(y)

			self.assertEqual(lib.ok_device_reset(), 0)

	def test_reduce(self):
		(m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
				single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 5 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				# allocate A, d, e
				A = lib.matrix(0, 0, 0, None, order)
				lib.matrix_calloc(A, m, n, order)
				A_py = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A_py += self.A_test
				A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_memcpy_ma(A, A_ptr, order)

				d = lib.vector(0, 0, None)
				lib.vector_calloc(d, m)
				d_py = np.zeros(m).astype(lib.pyfloat)
				d_ptr = d_py.ctypes.data_as(lib.ok_float_p)

				e = lib.vector(0, 0, None)
				lib.vector_calloc(e, n)
				e_py = np.zeros(n).astype(lib.pyfloat)
				e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

				x = lib.vector(0, 0, None)
				lib.vector_calloc(x, n)
				x_py = np.zeros(n).astype(lib.pyfloat)
				x_ptr = x_py.ctypes.data_as(lib.ok_float_p)

				y = lib.vector(0, 0, None)
				lib.vector_calloc(y, m)
				y_py = np.zeros(m).astype(lib.pyfloat)
				y_ptr = y_py.ctypes.data_as(lib.ok_float_p)

				x_py += np.random.rand(n)
				lib.vector_memcpy_va(x, x_ptr, 1)

				# min - reduce columns
				colmin = np.min(A_py, 0)
				lib.linalg_matrix_reduce_min(hdl, e, A, lib.enums.CblasLeft)
				lib.vector_memcpy_av(e_ptr, e, 1)
				self.assertTrue(np.linalg.norm(e_py - colmin) <=
								ATOLN + RTOL * np.linalg.norm(colmin))

				# min - reduce rows
				rowmin = np.min(A_py, 1)
				lib.linalg_matrix_reduce_min(hdl, d, A, lib.enums.CblasRight)
				lib.vector_memcpy_av(d_ptr, d, 1)
				self.assertTrue(np.linalg.norm(d_py - rowmin) <=
								ATOLM + RTOL * np.linalg.norm(rowmin))

				# max - reduce columns
				colmax = np.max(A_py, 0)
				lib.linalg_matrix_reduce_max(hdl, e, A, lib.enums.CblasLeft)
				lib.vector_memcpy_av(e_ptr, e, 1)
				self.assertTrue(np.linalg.norm(e_py - colmax) <=
								ATOLN + RTOL * np.linalg.norm(colmax))

				# max - reduce rows
				rowmax = np.max(A_py, 1)
				lib.linalg_matrix_reduce_max(hdl, d, A, lib.enums.CblasRight)
				lib.vector_memcpy_av(d_ptr, d, 1)
				self.assertTrue(np.linalg.norm(d_py - rowmax) <=
								ATOLM + RTOL * np.linalg.norm(rowmax))

				# indmin - reduce columns
				idx = lib.indvector(0, 0, None)
				lib.indvector_calloc(idx, n)
				inds = np.zeros(n).astype(c_size_t)
				inds_ptr = inds.ctypes.data_as(lib.c_size_t_p)
				lib.linalg_matrix_reduce_indmin(hdl, idx, e, A,
												lib.enums.CblasLeft)
				lib.indvector_memcpy_av(inds_ptr, idx, 1)
				lib.indvector_free(idx)
				calcmin = np.array([A_py[inds[i], i] for i in xrange(n)])
				colmin = np.min(A_py, 0)
				print colmin
				lib.vector_print(e)
				print calcmin - colmin
				self.assertTrue(np.linalg.norm(calcmin - colmin) <=
								ATOLN + RTOL * np.linalg.norm(colmin))

				# indmin - reduce rows
				lib.indvector_calloc(idx, m)
				inds = np.zeros(m).astype(c_size_t)
				inds_ptr = inds.ctypes.data_as(lib.c_size_t_p)
				lib.linalg_matrix_reduce_indmin(hdl, idx, d, A,
												lib.enums.CblasRight)
				lib.indvector_memcpy_av(inds_ptr, idx, 1)
				lib.indvector_free(idx)
				calcmin = np.array([A_py[i, inds[i]] for i in xrange(m)])
				rowmin = np.min(A_py, 1)
				print rowmin
				lib.vector_print(d)
				print calcmin - rowmin
				self.assertTrue(np.linalg.norm(calcmin - rowmin) <=
								ATOLM + RTOL * np.linalg.norm(rowmin))

				# free memory
				lib.matrix_free(A)
				lib.vector_free(d)
				lib.vector_free(e)
				lib.vector_free(x)
				lib.vector_free(y)

			self.assertEqual(lib.ok_device_reset(), 0)
