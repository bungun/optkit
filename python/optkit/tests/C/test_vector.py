import unittest
import os
import numpy as np
from optkit.libs import DenseLinsysLibs
from optkit.tests.defs import CONDITIONS, significant_digits, approx_compare

class VectorTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	@staticmethod
	def make_vec_triplet(lib, size_):
			a = lib.vector(0, 0, None)
			lib.vector_calloc(a, size_)
			a_py = np.zeros(size_).astype(lib.pyfloat)
			a_ptr = a_py.ctypes.data_as(lib.ok_float_p)
			return a, a_py, a_ptr

	def test_alloc(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			len_v = 10 + int(10 * np.random.rand())

			v = lib.vector(0, 0, None)
			self.assertEqual(v.size, 0)
			self.assertEqual(v.stride, 0)

			lib.vector_calloc(v, len_v)
			self.assertEqual(v.size, len_v)
			self.assertEqual(v.stride, 1)

			lib.vector_calloc(v, len_v)

			if not gpu:
				for i in xrange(v.size):
					self.assertEqual(v.data[i], 0)

			lib.vector_free(v)
			self.assertEqual(v.size, 0)
			self.assertEqual(v.stride, 0)

	def test_io(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			len_v = 10 + int(1000 * np.random.rand())

			v, v_py, v_ptr = self.make_vec_triplet(lib, len_v)

			w, w_py, w_ptr = self.make_vec_triplet(lib, len_v)

			# set_all
			set_val = 5
			lib.vector_set_all(v, set_val)

			# memcpy_av
			lib.vector_memcpy_av(v_ptr, v, 1)
			for i in xrange(len_v):
				self.assertEqual(v_py[i], set_val)

			# memcpy_va
			w_rand = np.random.rand(len_v)
			w_py[:] = w_rand[:]
			lib.vector_memcpy_va(w, w_ptr, 1)
			w_py *= 0
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(len_v):
				self.assertTrue(approx_compare(w_py[i], w_rand[i], 3))

			# memcpy_vv
			lib.vector_memcpy_vv(v, w)
			lib.vector_memcpy_av(v_ptr, v, 1)
			for i in xrange(len_v):
				self.assertTrue(approx_compare(v_py[i], w_rand[i], 3))

			# view_array
			if not gpu:
				u_rand = np.random.rand(len_v).astype(lib.pyfloat)
				u_ptr = u_rand.ctypes.data_as(lib.ok_float_p)
				u = lib.vector(0, 0, None)
				lib.vector_view_array(u, u_ptr, u_rand.size)
				for i in xrange(len_v):
					self.assertTrue(approx_compare(u_rand[i], u.data[i], 3))
				# DON'T FREE u, DATA OWNED BY PYTHON

			lib.vector_free(v)
			lib.vector_free(w)

	def test_subvector(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			len_v = 10 + int(10 * np.random.rand())

			v, v_py, v_ptr = self.make_vec_triplet(lib, len_v)

			offset_sub = 3
			len_sub = 3
			v_sub = lib.vector(0, 0, None)
			lib.vector_subvector(v_sub, v, offset_sub, len_sub)
			self.assertEqual(v_sub.size, 3)
			self.assertEqual(v_sub.stride, v.stride)
			v_sub_py = np.zeros(len_sub).astype(lib.pyfloat)
			v_sub_ptr = v_sub_py.ctypes.data_as(lib.ok_float_p)
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(v_sub_ptr, v_sub, 1)
			for i in xrange(len_sub):
				self.assertEqual(v_sub_py[i], v_py[i + offset_sub])

			lib.vector_free(v)

	def test_math(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT - 2 * lib.GPU

			val1 = 12 * np.random.rand()
			val2 = 5 * np.random.rand()
			len_v = 10 + int(1000 * np.random.rand())

			RTOL = 10**(-DIGITS)
			ATOL = RTOL * len_v**0.5

			v, v_py, v_ptr = self.make_vec_triplet(lib, len_v)
			w, w_py, w_ptr = self.make_vec_triplet(lib, len_v)

			# constant addition
			lib.vector_add_constant(v, val1)
			lib.vector_add_constant(w, val2)
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(v_py[i], val1, DIGITS))
				self.assertTrue(approx_compare(w_py[i], val2, DIGITS))

			# add two vectors
			lib.vector_add(v, w)
			val1 += val2
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(v_py[i], val1, DIGITS))
				self.assertTrue(approx_compare(w_py[i], val2, DIGITS))

			# subtract two vectors
			lib.vector_sub(w, v)
			val2 -= val1
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(v_py[i], val1, DIGITS))
				self.assertTrue(approx_compare(w_py[i], val2, DIGITS))

			# multiply two vectors
			lib.vector_mul(v, w)
			val1 *= val2
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(v_py[i], val1, DIGITS))
				self.assertTrue(approx_compare(w_py[i], val2, DIGITS))


			# vector scale
			scal = 3 * np.random.rand()
			val1 *= scal
			lib.vector_scale(v, scal)
			lib.vector_memcpy_av(v_ptr, v, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(v_py[i], val1, DIGITS))

			# make sure v is strictly positive
			val1 = 0.7 + np.random.rand()
			lib.vector_scale(v, 0)
			lib.vector_add_constant(v, val1)

			DIGITS -= 1

			# divide two vectors
			lib.vector_div(w, v)
			val2 /= float(val1)
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(v_py[i], val1, DIGITS))
				self.assertTrue(approx_compare(w_py[i], val2, DIGITS))

			# make w strictly negative
			w_max = w_py.max()
			val2 -= (w_max + 1)
			lib.vector_add_constant(w, -(w_max + 1))

			# vector abs
			lib.vector_abs(w)
			val2 = abs(val2)
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(w_py[i], val2, 3))

			# vector recip
			lib.vector_recip(w)
			val2 = 1. / val2
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(w_py[i], val2, 3))

			# vector sqrt
			lib.vector_sqrt(w)
			val2 **= 0.5
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(w_py[i], val2, 3))

			# vector pow
			pow_val = -2 + 4 * np.random.rand()
			lib.vector_pow(w, pow_val)
			val2 **= pow_val
			lib.vector_memcpy_av(w_ptr, w, 1)
			for i in xrange(v.size):
				self.assertTrue(approx_compare(w_py[i], val2, 3))

			# vector exp
			lib.vector_exp(w)
			val2 = np.exp(val2)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(val2 - w_py) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# min / max
			w_py *= 0
			w_py += np.random.rand(len(w_py))
			lib.vector_memcpy_va(w, w_ptr, 1)

			# vector argmin
			wargmin = lib.vector_indmin(w)
			self.assertTrue(w_py[wargmin] - w_py.min() <= 10**-3)

			# vector min
			wmin = lib.vector_min(w)
			self.assertTrue(wmin - w_py.min() <= 10**-3)

			# vector max
			wmax = lib.vector_max(w)
			self.assertTrue(wmax - w_py.max() <= 10**-3)

			lib.vector_free(v)
			lib.vector_free(w)