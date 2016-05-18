import os
import numpy as np
from optkit.libs.linsys import DenseLinsysLibs
from optkit.tests.C.base import OptkitCTestCase

class MatrixTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = DenseLinsysLibs()
		self.A_test = self.A_test_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_alloc(self):
		(m, n) = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				A = lib.matrix(0, 0, 0, None, order)
				self.assertEqual( A.size1, 0 )
				self.assertEqual( A.size2, 0 )
				self.assertEqual( A.ld, 0 )
				self.assertEqual( A.order, order )

				# calloc
				self.assertCall( lib.matrix_calloc(A, m, n, order) )
				self.register_var('A', A, lib.matrix_free)
				self.assertEqual( A.size1, m )
				self.assertEqual( A.size2, n )
				if order == lib.enums.CblasRowMajor:
					self.assertEqual( A.ld, n )
				else:
					self.assertEqual( A.ld, m )
				self.assertEqual( A.order, order )
				if not gpu:
					for i in xrange(m * n):
						self.assertEqual( A.data[i], 0 )
				# free
				self.free_var('A')
				self.assertCall( lib.ok_device_reset() )

	def test_io(self):
		(m, n) = self.shape
		A_rand = self.A_test
		x_rand = np.random.rand(n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 11 - 4 * lib.FLOAT - 1 * lib.GPU
			TOL = 10**(-DIGITS)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')

				# memcpy_am
				# set A_py to A_rand. overwrite A_py with zeros from A
				A_py += A_rand
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				for i in xrange(m):
					for j in xrange(n):
						self.assertEqual( A_py[i, j], 0)

				# memcpy_ma
				A_py += A_rand
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
				A_py *= 0
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, A_rand, TOL, TOL )

				# memcpy_mm
				Z, Z_py, Z_ptr = self.register_matrix(lib, m, n, order, 'Z')
				self.assertCall( lib.matrix_memcpy_mm(Z, A, order) )
				self.assertCall( lib.matrix_memcpy_am(Z_ptr, Z, order) )
				self.assertVecEqual( Z_py, A_py, TOL, TOL )

				# view_array
				if not gpu:
					A_py *= 0
					B = lib.matrix(0, 0, 0, None, order)
					self.assertCall( lib.matrix_view_array(B,
						A_rand.ctypes.data_as(lib.ok_float_p), m, n, order) )
					self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
					self.assertVecEqual( A_py, A_rand, TOL, TOL )

				# set_all
				val = 2
				A_rand *= 0
				A_rand += val
				self.assertCall( lib.matrix_set_all(A, val) )
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, A_rand, TOL, TOL )

				self.free_vars('A', 'Z')
				self.assertCall( lib.ok_device_reset() )

	def test_slicing(self):
		""" matrix slicing tests """
		(m, n) = self.shape
		A_rand = self.A_test
		x_rand = np.random.rand(n)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			TOL = 10**(-DIGITS)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				pyorder = 'C' if order == lib.enums.CblasRowMajor else 'F'

				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')

				# set A, A_py to A_rand
				A_py += A_rand
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				# submatrix
				m0 = m / 4
				n0 = n / 4
				msub = m / 2
				nsub = n / 2
				Asub = lib.matrix(0, 0, 0, None, order)
				Asub_py = np.zeros(
						(msub, nsub), order=pyorder).astype(lib.pyfloat)
				Asub_ptr = Asub_py.ctypes.data_as(lib.ok_float_p)

				self.assertCall( lib.matrix_submatrix(Asub, A, m0, n0, msub,
								 nsub) )
				self.assertCall( lib.matrix_memcpy_am(Asub_ptr, Asub, order) )
				A_py_sub = A_py[m0 : m0+msub, n0 : n0+nsub]
				self.assertVecEqual( Asub_py, A_py_sub, TOL, TOL )

				# row
				v = lib.vector(0, 0, None)
				v_py = np.zeros(n).astype(lib.pyfloat)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				self.assertCall( lib.matrix_row(v, A, m0) )
				self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
				self.assertVecEqual( A_py[m0, :], v_py, TOL, TOL )

				# column
				v_py = np.zeros(m).astype(lib.pyfloat)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				self.assertCall( lib.matrix_column(v, A, n0) )
				self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
				self.assertVecEqual( A_py[: , n0], v_py, TOL, TOL )

				# diagonal
				v_py = np.zeros(min(m, n)).astype(lib.pyfloat)
				v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
				self.assertCall( lib.matrix_diagonal(v, A) )
				self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
				self.assertVecEqual( np.diag(A_py), v_py, TOL, TOL )

				self.free_var('A')
				self.assertCall( lib.ok_device_reset() )

	def test_math(self):
		""" matrix math tests """
		(m, n) = self.shape
		A_rand = self.A_test

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLMN = RTOL * (m * n)**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				A, A_py, A_ptr = self.register_matrix(lib, m, n, order, 'A')

				# set A, A_py to A_rand
				A_py += A_rand
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				# scale: A = alpha * A
				alpha = np.random.rand()
				A_rand *= alpha
				self.assertCall( lib.matrix_scale(A, alpha) )
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, A_rand, ATOLMN, RTOL )

				# scale_left: A = diag(d) * A
				d, d_py, d_ptr = self.register_vector(lib, m, 'd')
				d_py[:] = np.random.rand(m)
				for i in xrange(m):
					A_rand[i, :] *= d_py[i]
				self.assertCall( lib.vector_memcpy_va(d, d_ptr, 1) )
				self.assertCall( lib.matrix_scale_left(A, d) )
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, A_rand, ATOLMN, RTOL )

				# scale_right: A = A * diag(e)
				e, e_py, e_ptr = self.register_vector(lib, n, 'e')
				e_py[:] = np.random.rand(n)
				for j in xrange(n):
					A_rand[:, j] *= e_py[j]
				self.assertCall( lib.vector_memcpy_va(e, e_ptr, 1) )
				self.assertCall( lib.matrix_scale_right(A, e) )
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, A_rand, ATOLMN, RTOL )

				# abs: A_ij = abs(A_ij)
				A_rand -= (A_rand.max() - A_rand.min()) / 2
				A_py *= 0
				A_py += A_rand
				A_rand = np.abs(A_rand)
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )
				self.assertCall( lib.matrix_abs(A) )
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, A_rand, ATOLMN, RTOL )

				# pow
				p = 3 * np.random.rand()
				A_rand **= p
				self.assertCall( lib.matrix_pow(A, p) )
				self.assertCall( lib.matrix_memcpy_am(A_ptr, A, order) )
				self.assertVecEqual( A_py, A_rand, ATOLMN, RTOL )

				self.free_vars('d', 'e', 'A')
				self.assertCall( lib.ok_device_reset() )