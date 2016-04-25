import unittest
import os
import numpy as np
from ctypes import c_int, byref, c_void_p
from optkit.libs import ProxLibs
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.defs OptkitTestCase
from optkit.tests.C.base import OptkitCTestCase

class ProxLibsTestCase(OptkitTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProxLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue(any(libs))

	def test_lib_types(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertTrue('function' in dir(lib))
			self.assertTrue('function_p' in dir(lib))
			self.assertTrue('function_vector' in dir(lib))
			self.assertTrue('function_vector_p' in dir(lib))

	def test_version(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			major = c_int()
			minor = c_int()
			change = c_int()
			status = c_int()

			lib.optkit_version(byref(major), byref(minor), byref(change),
							   byref(status))

			version = self.version_string(major.value, minor.value,
										  change.value, status.value)

			self.assertNotEqual(version, '0.0.0')
			if VERBOSE_TEST:
				print("proxlib version", version)

class ProxTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProxLibs()
		self.shape = DEFAULT_SHAPE
		self.scalefactor = 5

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()

	@staticmethod
	def make_prox_triplet(lib, size_):
		f = lib.function_vector(0, None)
		lib.function_vector_calloc(f, size_)
		f_py = np.zeros(size_, dtype=lib.function)
		f_ptr = f_py.ctypes.data_as(lib.function_p)
		return f, f_py, f_ptr

	def test_alloc(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			# calloc
			f = lib.function_vector(0, None)
			self.assertEqual(f.size, 0)

			lib.function_vector_calloc(f, m)
			self.register_var('f', f, lib.function_vector_free)
			self.assertEqual(f.size, m)

			# free
			self.free_var('f')
			self.assertEqual(f.size, 0)

			self.assertEqual(lib.ok_device_reset(), 0)

	def test_io(self):
		m, n = self.shape
		scal = self.scalefactor
		a = scal * np.random.rand()
		b = np.random.rand()
		c = np.random.rand()
		d = np.random.rand()
		e = np.random.rand()

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			f, f_py, f_ptr = self.make_prox_triplet(lib, m)
			self.register_var('f', f, lib.function_vector_free)

			# initialize to default values
			hlast = 0
			alast = 1.
			blast = 0.
			clast = 1.
			dlast = 0.
			elast = 0.

			for hkey, hval in lib.function_enums.dict.items():
				if VERBOSE_TEST:
					print hkey

				for i in xrange(m):
					f_py[i] = lib.function(hval, a, b, c, d, e)

				f_list = [lib.function(*f_) for f_ in f_py]
				fh = [f_.h - hval for f_ in f_list]
				fa = [f_.a - a for f_ in f_list]
				fb = [f_.b - b for f_ in f_list]
				fc = [f_.c - c for f_ in f_list]
				fd = [f_.d - d for f_ in f_list]
				fe = [f_.e - e for f_ in f_list]
				self.assertTrue( np.linalg.norm(fh) <= ATOLM )
				self.assertTrue( np.linalg.norm(fa) <= ATOLM )
				self.assertTrue( np.linalg.norm(fb) <= ATOLM )
				self.assertTrue( np.linalg.norm(fc) <= ATOLM )
				self.assertTrue( np.linalg.norm(fd) <= ATOLM )
				self.assertTrue( np.linalg.norm(fe) <= ATOLM )
				# memcpy af
				lib.function_vector_memcpy_av(f_ptr, f)
				f_list = [lib.function(*f_) for f_ in f_py]
				fh = [f_.h - hlast for f_ in f_list]
				fa = [f_.a - alast for f_ in f_list]
				fb = [f_.b - blast for f_ in f_list]
				fc = [f_.c - clast for f_ in f_list]
				fd = [f_.d - dlast for f_ in f_list]
				fe = [f_.e - elast for f_ in f_list]
				self.assertTrue( np.linalg.norm(fh) <= ATOLM )
				self.assertTrue( np.linalg.norm(fa) <= ATOLM )
				self.assertTrue( np.linalg.norm(fb) <= ATOLM )
				self.assertTrue( np.linalg.norm(fc) <= ATOLM )
				self.assertTrue( np.linalg.norm(fd) <= ATOLM )
				self.assertTrue( np.linalg.norm(fe) <= ATOLM )


				# memcpy fa
				for i in xrange(m):
					f_py[i] = lib.function(hval, a, b, c, d, e)

				lib.function_vector_memcpy_va(f, f_ptr)
				lib.function_vector_memcpy_av(f_ptr, f)

				f_list = [lib.function(*f_) for f_ in f_py]
				fh = [f_.h - hval for f_ in f_list]
				fa = [f_.a - a for f_ in f_list]
				fb = [f_.b - b for f_ in f_list]
				fc = [f_.c - c for f_ in f_list]
				fd = [f_.d - d for f_ in f_list]
				fe = [f_.e - e for f_ in f_list]
				self.assertTrue( np.linalg.norm(fh) <= ATOLM )
				self.assertTrue( np.linalg.norm(fa) <= ATOLM )
				self.assertTrue( np.linalg.norm(fb) <= ATOLM )
				self.assertTrue( np.linalg.norm(fc) <= ATOLM )
				self.assertTrue( np.linalg.norm(fd) <= ATOLM )
				self.assertTrue( np.linalg.norm(fe) <= ATOLM )



				hlast = hval
				alast = a
				blast = b
				clast = c
				dlast = d
				elast = e

			self.free_var('f')

	def test_math(self):
		m, n = self.shape
		a = 1 + np.random.rand(m)
		b = 1 + np.random.rand(m)
		c = 1 + np.random.rand(m)
		d = 1 + np.random.rand(m)
		e = 1 + np.random.rand(m)
		# (add 1 above to make sure no divide by zero below)

		hval = 0

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			f, f_py, f_ptr = self.make_prox_triplet(lib, m)
			self.register_var('f', f, lib.function_vector_free)

			v = lib.vector(0, 0, None)
			lib.vector_calloc(v, m)
			self.register_var('v', v, lib.vector_free)

			v_py = np.zeros(m).astype(lib.pyfloat)
			v_ptr = v_py.ctypes.data_as(lib.ok_float_p)

			for i in xrange(m):
				f_py[i] = lib.function(hval, a[i], b[i], c[i], d[i], e[i])
			lib.function_vector_memcpy_va(f, f_ptr)
			v_py[:] = np.random.rand(m)
			lib.vector_memcpy_va(v, v_ptr, 1)

			# mul
			lib.function_vector_mul(f, v)
			lib.function_vector_memcpy_av(f_ptr, f)
			for i in xrange(m):
				a[i] *= v_py[i]
				d[i] *= v_py[i]
				e[i] *= v_py[i]
			f_list = [lib.function(*f_) for f_ in f_py]
			fh = [f_.h - hval for f_ in f_list]
			fa = [f_.a for f_ in f_list]
			fb = [f_.b for f_ in f_list]
			fc = [f_.c for f_ in f_list]
			fd = [f_.d for f_ in f_list]
			fe = [f_.e for f_ in f_list]
			self.assertTrue( np.linalg.norm(fh) <= ATOLM )
			self.assertTrue( np.linalg.norm(fa - a) <=
							 ATOLM + RTOL * np.linalg.norm(a) )
			self.assertTrue( np.linalg.norm(fb - b) <=
							 ATOLM + RTOL * np.linalg.norm(b) )
			self.assertTrue( np.linalg.norm(fc - c) <=
							 ATOLM + RTOL * np.linalg.norm(c) )
			self.assertTrue( np.linalg.norm(fd - d) <=
							 ATOLM + RTOL * np.linalg.norm(d) )
			self.assertTrue( np.linalg.norm(fe - e) <=
							 ATOLM + RTOL * np.linalg.norm(e) )

			# div
			lib.function_vector_div(f, v)
			lib.function_vector_memcpy_av(f_ptr, f)
			for i in xrange(m):
				a[i] /= v_py[i]
				d[i] /= v_py[i]
				e[i] /= v_py[i]
			f_list = [lib.function(*f_) for f_ in f_py]
			fh = [f_.h - hval for f_ in f_list]
			fa = [f_.a for f_ in f_list]
			fb = [f_.b for f_ in f_list]
			fc = [f_.c for f_ in f_list]
			fd = [f_.d for f_ in f_list]
			fe = [f_.e for f_ in f_list]
			self.assertTrue( np.linalg.norm(fh) <= ATOLM )
			self.assertTrue( np.linalg.norm(fa - a) <=
							 ATOLM + RTOL * np.linalg.norm(a) )
			self.assertTrue( np.linalg.norm(fb - b) <=
							 ATOLM + RTOL * np.linalg.norm(b) )
			self.assertTrue( np.linalg.norm(fc - c) <=
							 ATOLM + RTOL * np.linalg.norm(c) )
			self.assertTrue( np.linalg.norm(fd - d) <=
							 ATOLM + RTOL * np.linalg.norm(d) )
			self.assertTrue( np.linalg.norm(fe - e) <=
							 ATOLM + RTOL * np.linalg.norm(e) )

			self.free_var('f')
			self.free_var('v')

	def test_eval(self):
		m, n = self.shape
		scal = self.scalefactor
		a = 10 * np.random.rand(m)
		b = np.random.rand(m)
		c = np.random.rand(m)
		d = np.random.rand(m)
		e = np.random.rand(m)
		x_rand = np.random.rand(m)

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			f, f_py, f_ptr = self.make_prox_triplet(lib, m)
			self.register_var('f', f, lib.function_vector_free)

			x = lib.vector(0, 0, None)
			x_py = np.zeros(m).astype(lib.pyfloat)
			x_ptr = x_py.ctypes.data_as(lib.ok_float_p)
			lib.vector_calloc(x, m)
			self.register_var('x', x, lib.vector_free)

			xout = lib.vector(0, 0, None)
			xout_py = np.zeros(m).astype(lib.pyfloat)
			xout_ptr = xout_py.ctypes.data_as(lib.ok_float_p)
			lib.vector_calloc(xout, m)
			self.register_var('xout', xout, lib.vector_free)

			# populate
			x_py += x_rand
			lib.vector_memcpy_va(x, x_ptr, 1)

			for hkey, hval in lib.function_enums.dict.items():
				if VERBOSE_TEST:
					print hkey

				# avoid domain errors with randomly generated data
				if 'Log' in hkey or 'Exp' in hkey or 'Entr' in hkey:
					continue

				for i in xrange(m):
					f_py[i] = lib.function(hval, a[i], b[i], c[i], d[i], e[i])
				lib.function_vector_memcpy_va(f, f_ptr)

				# function evaluation
				f_list = [lib.function(*f_) for f_ in f_py]
				funcval_py = func_eval_python(f_list, x_rand)
				funcval_c = lib.FuncEvalVector(f, x)
				if funcval_c in (np.inf, np.nan):
					self.assertTrue( 1 )
				else:
					self.assertTrue( np.abs(funcval_py - funcval_c) <=
									 ATOLM + RTOL * np.abs(funcval_c) )

				# proximal operator evaluation, random rho
				rho = 5 * np.random.rand()
				prox_py = prox_eval_python(f_list, rho, x_rand)
				lib.ProxEvalVector(f, rho, x, xout)
				lib.vector_memcpy_av(xout_ptr, xout, 1)
				self.assertTrue( np.linalg.norm(xout_py - prox_py) <=
								 ATOLM + RTOL * np.linalg.norm(prox_py) )

			self.free_var('f')
			self.free_var('x')
			self.free_var('xout')