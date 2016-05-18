import os
import numpy as np
from ctypes import c_size_t
from optkit.libs.linsys import DenseLinsysLibs
from optkit.tests.C.base import OptkitCTestCase

class VectorTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_alloc(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			len_v = 10 + int(10 * np.random.rand())

			v = lib.vector(0, 0, None)
			self.assertEqual( v.size, 0 )
			self.assertEqual( v.stride, 0 )

			self.assertCall( lib.vector_calloc(v, len_v) )
			self.register_var('v', v, lib.vector_free)

			self.assertEqual( v.size, len_v )
			self.assertEqual( v.stride, 1 )

			if not gpu:
				for i in xrange(v.size):
					self.assertEqual( v.data[i], 0 )

			self.free_var('v')
			self.assertEqual( v.size, 0 )
			self.assertEqual( v.stride, 0 )

	def test_io(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			len_v = 10 + int(1000 * np.random.rand())
			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOL = RTOL * len_v**0.5

			v, v_py, v_ptr = self.register_vector(lib, len_v, 'v')
			w, w_py, w_ptr = self.register_vector(lib, len_v, 'w')

			# set_all
			set_val = 5
			self.assertCall( lib.vector_set_all(v, set_val) )

			# memcpy_av
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			for i in xrange(len_v):
				self.assertEqual(v_py[i], set_val)

			# memcpy_va
			w_rand = np.random.rand(len_v)
			w_py[:] = w_rand[:]
			self.assertCall( lib.vector_memcpy_va(w, w_ptr, 1) )
			w_py *= 0
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( w_py, w_rand, ATOL, RTOL )

			# memcpy_vv
			self.assertCall( lib.vector_memcpy_vv(v, w) )
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertVecEqual( v_py, w_rand, ATOL, RTOL )

			# view_array
			if not gpu:
				u_rand = np.random.rand(len_v).astype(lib.pyfloat)
				u_ptr = u_rand.ctypes.data_as(lib.ok_float_p)
				u = lib.vector(0, 0, None)
				self.assertCall( lib.vector_view_array(u, u_ptr,
													   u_rand.size) )
				self.assertCall( lib.vector_memcpy_av(v_ptr, u, 1) )
	 			self.assertVecEqual( v_py, u_rand, ATOL, RTOL )

				# DON'T FREE u, DATA OWNED BY PYTHON

			self.free_vars('v', 'w')
			self.assertCall( lib.ok_device_reset() )

	def test_subvector(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			len_v = 10 + int(10 * np.random.rand())

			v, v_py, v_ptr = self.register_vector(lib, len_v, 'v')

			offset_sub = 3
			len_sub = 3
			v_sub = lib.vector(0, 0, None)
			self.assertCall( lib.vector_subvector(v_sub, v, offset_sub,
												  len_sub) )
			self.assertEqual( v_sub.size, 3 )
			self.assertEqual( v_sub.stride, v.stride )
			v_sub_py = np.zeros(len_sub).astype(lib.pyfloat)
			v_sub_ptr = v_sub_py.ctypes.data_as(lib.ok_float_p)
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertCall( lib.vector_memcpy_av(v_sub_ptr, v_sub, 1) )
			for i in xrange(len_sub):
				self.assertEqual( v_sub_py[i], v_py[i + offset_sub] )

			self.free_var('v')
			self.assertCall( lib.ok_device_reset() )

	def test_math(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			val1 = 12 * np.random.rand()
			val2 = 5 * np.random.rand()
			len_v = 10 + int(1000 * np.random.rand())
			# len_v = 10 + int(10 * np.random.rand())

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOL = RTOL * len_v**0.5

			v, v_py, v_ptr = self.register_vector(lib, len_v, 'v')
			w, w_py, w_ptr = self.register_vector(lib, len_v, 'w')

			# constant addition
			self.assertCall( lib.vector_add_constant(v, val1) )
			self.assertCall( lib.vector_add_constant(w, val2) )
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( v_py, val1, ATOL, RTOL )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# add two vectors
			self.assertCall( lib.vector_add(v, w) )
			val1 += val2
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( v_py, val1, ATOL, RTOL )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# subtract two vectors
			self.assertCall( lib.vector_sub(w, v) )
			val2 -= val1
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( v_py, val1, ATOL, RTOL )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# multiply two vectors
			self.assertCall( lib.vector_mul(v, w) )
			val1 *= val2
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( v_py, val1, ATOL, RTOL )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# vector scale
			scal = 3 * np.random.rand()
			val1 *= scal
			self.assertCall( lib.vector_scale(v, scal) )
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertVecEqual( v_py, val1, ATOL, RTOL )

			# make sure v is strictly positive
			val1 = 0.7 + np.random.rand()
			self.assertCall( lib.vector_scale(v, 0) )
			self.assertCall( lib.vector_add_constant(v, val1) )

			# divide two vectors
			self.assertCall( lib.vector_div(w, v) )
			val2 /= float(val1)
			self.assertCall( lib.vector_memcpy_av(v_ptr, v, 1) )
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( v_py, val1, ATOL, RTOL )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# make w strictly negative
			w_max = w_py.max()
			val2 -= (w_max + 1)
			self.assertCall( lib.vector_add_constant(w, -(w_max + 1)) )

			# vector abs
			self.assertCall( lib.vector_abs(w) )
			val2 = abs(val2)
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# vector recip
			self.assertCall( lib.vector_recip(w) )
			val2 = 1. / val2
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# vector sqrt
			self.assertCall( lib.vector_sqrt(w) )
			val2 **= 0.5
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# vector pow
			pow_val = -2 + 4 * np.random.rand()
			self.assertCall( lib.vector_pow(w, pow_val) )
			val2 **= pow_val
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( w_py, val2, ATOL, RTOL )

			# vector exp
			self.assertCall( lib.vector_exp(w) )
			val2 = np.exp(val2)
			self.assertCall( lib.vector_memcpy_av(w_ptr, w, 1) )
			self.assertVecEqual( val2, w_py, ATOL, RTOL )

			# min / max
			w_py *= 0
			w_py += np.random.rand(len(w_py))
			self.assertCall( lib.vector_memcpy_va(w, w_ptr, 1) )

			# vector argmin
			wargmin = np.zeros(1).astype(c_size_t)
			wargmin_p = wargmin.ctypes.data_as(lib.c_size_t_p)
			self.assertCall( lib.vector_indmin(w, wargmin_p) )
			self.assertScalarEqual( w_py[wargmin[0]], w_py.min(), RTOL )

			# # vector min
			wmin = np.zeros(1).astype(lib.pyfloat)
			wmin_p = wmin.ctypes.data_as(lib.ok_float_p)
			self.assertCall( lib.vector_min(w, wmin_p) )
			self.assertScalarEqual( wmin[0], w_py.min(), RTOL )

			# # vector max
			wmax = wmin
			wmax_p = wmin_p
			self.assertCall( lib.vector_max(w, wmax_p) )
			self.assertScalarEqual( wmax[0], w_py.max(), RTOL )

			self.free_vars('v', 'w')
			self.assertCall( lib.ok_device_reset() )

	def test_indvector_math(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			val1 = 12 * np.random.rand()
			val2 = 5 * np.random.rand()
			len_v = 10 + int(1000 * np.random.rand())
			# len_v = 10 + int(10 * np.random.rand())

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)

			w, w_py, w_ptr = self.register_indvector(lib, len_v, 'w')

			# min / max
			w_py *= 0
			w_py += (30 * np.random.rand(len(w_py))).astype(w_py.dtype)
			self.assertCall( lib.indvector_memcpy_va(w, w_ptr, 1) )

			# vector argmin
			wargmin = np.zeros(1).astype(c_size_t)
			wargmin_p = wargmin.ctypes.data_as(lib.c_size_t_p)
			self.assertCall( lib.indvector_indmin(w, wargmin_p) )
			self.assertScalarEqual( w_py[wargmin[0]], w_py.min(), RTOL )

			# # vector min
			wmin = wargmin
			wmin_p = wargmin_p
			self.assertCall( lib.indvector_min(w, wmin_p) )
			self.assertScalarEqual( wmin[0], w_py.min(), RTOL )

			# vector max
			wmax = wmin
			wmax_p = wmin_p
			self.assertCall( lib.indvector_max(w, wmax_p) )
			self.assertScalarEqual( wmax[0], w_py.max(), RTOL )

			self.free_var('w')
			self.assertCall( lib.ok_device_reset() )
