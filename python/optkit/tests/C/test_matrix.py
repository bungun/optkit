import unittest
import os
import numpy as np
from optkit.libs import DenseLinsysLibs
from optkit.tests.defs import CONDITIONS

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
		self.shape = (150, 250)
		self.A_test = np.random.rand(*self.shape)

	@staticmethod
	def make_mat_triplet(lib, shape, pytype, rowmajor=True):
		order = 101 if rowmajor else 102
		pyorder = 'C' if rowmajor else 'F'
		A = lib.matrix(0, 0, 0, None, order)
		lib.matrix_calloc(A, shape[0], shape[1], order)
		A_py = np.zeros(shape, order=pyorder).astype(pytype)
		A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
		return A, A_py, A_ptr

	def test_alloc(self):
		(m, n) = self.shape

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(single_precision=single_precision,
									  gpu=gpu)
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
			lib = self.dense_libs.get(single_precision=single_precision,
									  gpu=gpu)
			if lib is None:
				continue

			pytype = np.float32 if single_precision else np.float64

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n), pytype,
													   rowmajor)

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
				self.assertTrue(np.allclose(A_py, A_rand))

				# memcpy_mm
				Z, Z_py, Z_ptr = self.make_mat_triplet(lib, (m, n), pytype,
													   rowmajor)
				lib.matrix_memcpy_mm(Z, A, order)
				lib.matrix_memcpy_am(Z_ptr, Z, order)
				self.assertTrue(np.allclose(Z_py, A_py))

				# view_array
				if not gpu:
					A_py *= 0
					B = lib.matrix(0, 0, 0, None, order)
					lib.matrix_view_array(B,
						A_rand.ctypes.data_as(lib.ok_float_p), m, n, order)
					lib.matrix_memcpy_am(A_ptr, A, order)
					self.assertTrue(np.allclose(A_py, A_rand))

				# set_all
				val = 2
				A_rand *= 0
				A_rand += val
				lib.matrix_set_all(A, val)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand))

			self.assertEqual(lib.ok_device_reset(), 0)

	def test_slicing(self):
		(m, n) = self.shape
		A_rand = self.A_test
		x_rand = np.random.rand(n)

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(single_precision=single_precision,
									  gpu=gpu)
			if lib is None:
				continue

			pytype = np.float32 if single_precision else np.float64

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n), pytype,
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
				Asub_py = np.zeros((msub, nsub), order=pyorder).astype(pytype)
				Asub_ptr = Asub_py.ctypes.data_as(lib.ok_float_p)

				lib.matrix_submatrix(Asub, A, m0, n0, msub, nsub)
				lib.matrix_memcpy_am(Asub_ptr, Asub, order)
				self.assertTrue(np.allclose(A_py[m0 : m0+msub, n0 : n0+nsub],
											Asub_py))

				# row
				v = lib.vector(0, 0, None)
				v_py = np.zeros(n).astype(pytype)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_row(v, A, m0)
				lib.vector_memcpy_av(v_ptr, v, 1)
				self.assertTrue(np.allclose(A_py[m0, :], v_py))

				# column
				v_py = np.zeros(m).astype(pytype)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_column(v, A, n0)
				lib.vector_memcpy_av(v_ptr, v, 1)
				self.assertTrue(np.allclose(A_py[: , n0], v_py))

				# diagonal
				v_py = np.zeros(min(m, n)).astype(pytype)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				lib.matrix_diagonal(v, A)
				lib.vector_memcpy_av(v_ptr, v, 1)
				self.assertTrue(np.allclose(np.diag(A_py), v_py))

	def test_math(self):
		(m, n) = self.shape
		A_rand = self.A_test

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(single_precision=single_precision,
									  gpu=gpu)
			if lib is None:
				continue

			pytype = np.float32 if single_precision else np.float64

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A, A_py, A_ptr = self.make_mat_triplet(lib, (m, n), pytype,
								   					   rowmajor)

				# set A, A_py to A_rand
				A_py += A_rand
				lib.matrix_memcpy_ma(A, A_ptr, order)

				# scale: A = alpha * A
				alpha = np.random.rand()
				A_rand *= alpha
				lib.matrix_scale(A, alpha)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand))

				# scale_left: A = diag(d) * A
				d = lib.vector(0, 0, None)
				lib.vector_calloc(d, m)
				d_py = np.zeros(m).astype(pytype)
				d_ptr = d_py.ctypes.data_as(lib.ok_float_p)
				d_py[:] = np.random.rand(m)
				for i in xrange(m):
					A_rand[i, :] *= d_py[i]
				lib.vector_memcpy_va(d, d_ptr, 1)
				lib.matrix_scale_left(A, d)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand))

				# scale_right: A = A * diag(e)
				e = lib.vector(0, 0, None)
				lib.vector_calloc(e, n)
				e_py = np.zeros(n).astype(pytype)
				e_ptr = e_py.ctypes.data_as(lib.ok_float_p)
				e_py[:] = np.random.rand(n)
				for j in xrange(n):
					A_rand[:, j] *= e_py[j]
				lib.vector_memcpy_va(e, e_ptr, 1)
				lib.matrix_scale_right(A, e)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand))

				# abs: A_ij = abs(A_ij)
				A_rand -= (A_rand.max() - A_rand.min()) / 2
				A_py *= 0
				A_py += A_rand
				A_rand = np.abs(A_rand)
				lib.matrix_memcpy_ma(A, A_ptr, order)
				lib.matrix_abs(A)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand))

				# pow
				p = 3 * np.random.rand()
				A_rand **= p
				lib.matrix_pow(A, p)
				lib.matrix_memcpy_am(A_ptr, A, order)
				self.assertTrue(np.allclose(A_py, A_rand))