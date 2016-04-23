import unittest
import os
import numpy as np
from ctypes import c_size_t
from optkit.libs import DenseLinsysLibs
from optkit.tests.defs import CONDITIONS, significant_digits, approx_compare
from optkit.tests.C.base import OptkitCTestCase

class VectorTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def tearDown(self):
		self.free_all_vars()

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
			self.register_var('v', v, lib.vector_free)

			self.assertEqual(v.size, len_v)
			self.assertEqual(v.stride, 1)

			lib.vector_calloc(v, len_v)

			if not gpu:
				for i in xrange(v.size):
					self.assertEqual(v.data[i], 0)

			self.free_var('v')
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
			self.register_var('v', v, lib.vector_free)
			self.register_var('w', w, lib.vector_free)


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

			self.free_var('v')
			self.free_var('w')

	def test_subvector(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			len_v = 10 + int(10 * np.random.rand())

			v, v_py, v_ptr = self.make_vec_triplet(lib, len_v)
			self.register_var('v', v, lib.vector_free)

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

			self.free_var('v')

	def test_math(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU

			val1 = 12 * np.random.rand()
			val2 = 5 * np.random.rand()
			len_v = 10 + int(1000 * np.random.rand())
			# len_v = 10 + int(10 * np.random.rand())

			RTOL = 10**(-DIGITS)
			ATOL = RTOL * len_v**0.5

			v = lib.vector(0, 0, None)
			lib.vector_calloc(v, len_v)
			self.register_var('v', v, lib.vector_free)

			v_py = np.zeros(len_v).astype(lib.pyfloat)
			v_ptr = v_py.ctypes.data_as(lib.ok_float_p)

			w = lib.vector(0, 0, None)
			lib.vector_calloc(w, len_v)
			self.register_var('w', w, lib.vector_free)

			w_py = np.zeros(len_v).astype(lib.pyfloat)
			w_ptr = w_py.ctypes.data_as(lib.ok_float_p)

			# constant addition
			lib.vector_add_constant(v, val1)
			lib.vector_add_constant(w, val2)
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(v_py - val1) <=
				ATOL + RTOL * np.linalg.norm(v_py))
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# add two vectors
			lib.vector_add(v, w)
			val1 += val2
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(v_py - val1) <=
				ATOL + RTOL * np.linalg.norm(v_py))
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# subtract two vectors
			lib.vector_sub(w, v)
			val2 -= val1
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(v_py - val1) <=
				ATOL + RTOL * np.linalg.norm(v_py))
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# multiply two vectors
			lib.vector_mul(v, w)
			val1 *= val2
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(v_py - val1) <=
				ATOL + RTOL * np.linalg.norm(v_py))
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# vector scale
			scal = 3 * np.random.rand()
			val1 *= scal
			lib.vector_scale(v, scal)
			lib.vector_memcpy_av(v_ptr, v, 1)
			self.assertTrue(np.linalg.norm(v_py - val1) <=
				ATOL + RTOL * np.linalg.norm(v_py))

			# make sure v is strictly positive
			val1 = 0.7 + np.random.rand()
			lib.vector_scale(v, 0)
			lib.vector_add_constant(v, val1)

			# divide two vectors
			lib.vector_div(w, v)
			val2 /= float(val1)
			lib.vector_memcpy_av(v_ptr, v, 1)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(v_py - val1) <=
				ATOL + RTOL * np.linalg.norm(v_py))
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# make w strictly negative
			w_max = w_py.max()
			val2 -= (w_max + 1)
			lib.vector_add_constant(w, -(w_max + 1))

			# vector abs
			lib.vector_abs(w)
			val2 = abs(val2)
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# vector recip
			lib.vector_recip(w)
			val2 = 1. / val2
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# vector sqrt
			lib.vector_sqrt(w)
			val2 **= 0.5
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# vector pow
			pow_val = -2 + 4 * np.random.rand()
			lib.vector_pow(w, pow_val)
			val2 **= pow_val
			lib.vector_memcpy_av(w_ptr, w, 1)
			self.assertTrue(np.linalg.norm(w_py - val2) <=
				ATOL + RTOL * np.linalg.norm(w_py))

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
			self.assertTrue(w_py[wargmin] - w_py.min() <=
				ATOL + RTOL * np.linalg.norm(w_py))


			# vector min
			wmin = lib.vector_min(w)


			print wmin
			print w_py.min()

			self.assertTrue(wmin - w_py.min() <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# vector max
			wmax = lib.vector_max(w)
			self.assertTrue(wmax - w_py.max() <=
				ATOL + RTOL * np.linalg.norm(w_py))

			self.free_var('v')
			self.free_var('w')

	def test_indvector_math(self):
		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU

			val1 = 12 * np.random.rand()
			val2 = 5 * np.random.rand()
			len_v = 10 + int(1000 * np.random.rand())
			# len_v = 10 + int(10 * np.random.rand())

			RTOL = 10**(-DIGITS)
			ATOL = RTOL * len_v**0.5

			w = lib.indvector(0, 0, None)
			lib.indvector_calloc(w, len_v)
			self.register_var('w', w, lib.indvector_free)

			w_py = np.zeros(len_v).astype(c_size_t)
			w_ptr = w_py.ctypes.data_as(lib.c_size_t_p)

			# min / max
			w_py *= 0
			w_py += (30 * np.random.rand(len(w_py))).astype(w_py.dtype)
			lib.indvector_memcpy_va(w, w_ptr, 1)

			# vector argmin
			wargmin = lib.indvector_indmin(w)
			self.assertTrue(w_py[wargmin] - w_py.min() <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# vector min
			wmin = lib.indvector_min(w)
			self.assertTrue(wmin - w_py.min() <=
				ATOL + RTOL * np.linalg.norm(w_py))

			# vector max
			wmax = lib.indvector_max(w)
			self.assertTrue(wmax - w_py.max() <=
				ATOL + RTOL * np.linalg.norm(w_py))

			self.free_var('w')