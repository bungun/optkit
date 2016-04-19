import unittest
import os
import numpy as np
from ctypes import c_int, byref, c_void_p
from optkit.libs import DenseLinsysLibs, ProxLibs
from optkit.tests.defs import VERBOSE_TEST, CONDITIONS, version_string, \
							  DEFAULT_SHAPE, significant_digits
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.C.base import OptkitCTestCase

class ProxLibsTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.prox_libs = ProxLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		dlibs = []
		pxlibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(
					single_precision=single_precision, gpu=gpu))
			pxlibs.append(self.prox_libs.get(
					dlibs[-1], single_precision=single_precision, gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(pxlibs))

	def test_lib_types(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.prox_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertTrue('function' in dir(lib))
			self.assertTrue('function_p' in dir(lib))
			self.assertTrue('function_vector' in dir(lib))
			self.assertTrue('function_vector_p' in dir(lib))

	def test_version(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.prox_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			major = c_int()
			minor = c_int()
			change = c_int()
			status = c_int()

			dlib.denselib_version(byref(major), byref(minor), byref(change),
								  byref(status))

			dversion = version_string(major.value, minor.value, change.value,
									  status.value)

			self.assertNotEqual(dversion, '0.0.0')

			lib.proxlib_version(byref(major), byref(minor), byref(change),
								byref(status))

			pxversion = version_string(major.value, minor.value, change.value,
									   status.value)

			self.assertNotEqual(pxversion, '0.0.0')
			self.assertEqual(pxversion, dversion)
			if VERBOSE_TEST:
				print("proxlib version", pxversion)

class ProxTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.prox_libs = ProxLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = DEFAULT_SHAPE
		self.scalefactor = 5

	@staticmethod
	def make_prox_triplet(lib, size_):
		f = lib.function_vector(0, None)
		lib.function_vector_calloc(f, size_)
		f_py = np.zeros(size_, dtype=lib.function)
		f_ptr = f_py.ctypes.data_as(lib.function_p)
		return f, f_py, f_ptr

	def test_alloc(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.prox_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
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

			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_io(self):
		m, n = self.shape
		scal = self.scalefactor
		a = scal * np.random.rand()
		b = np.random.rand()
		c = np.random.rand()
		d = np.random.rand()
		e = np.random.rand()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.prox_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if lib.FLOAT else 7

			f, f_py, f_ptr = self.make_prox_triplet(lib, m)
			self.register_var('f', f, lib.function_vector_free)

			# initialize to default values
			hlast = 0
			alast = 1.
			blast = 0.
			clast = 1.
			dlast = 0.
			elast = 0.

			for hkey, hval in lib.enums.dict.items():
				if VERBOSE_TEST:
					print hkey

				for i in xrange(m):
					f_py[i] = lib.function(hval, a, b, c, d, e)

				f_list = [lib.function(*f_) for f_ in f_py]
				for i in xrange(m):
					self.assertAlmostEqual(f_list[i].h, hval, DIGITS)
					self.assertAlmostEqual(f_list[i].a, a, DIGITS)
					self.assertAlmostEqual(f_list[i].b, b, DIGITS)
					self.assertAlmostEqual(f_list[i].c, c, DIGITS)
					self.assertAlmostEqual(f_list[i].d, d, DIGITS)
					self.assertAlmostEqual(f_list[i].e, e, DIGITS)

				# memcpy af
				lib.function_vector_memcpy_av(f_ptr, f)
				f_list = [lib.function(*f_) for f_ in f_py]
				for i in xrange(m):
					self.assertAlmostEqual(f_list[i].h, hlast, DIGITS)
					self.assertAlmostEqual(f_list[i].a, alast, DIGITS)
					self.assertAlmostEqual(f_list[i].b, blast, DIGITS)
					self.assertAlmostEqual(f_list[i].c, clast, DIGITS)
					self.assertAlmostEqual(f_list[i].d, dlast, DIGITS)
					self.assertAlmostEqual(f_list[i].e, elast, DIGITS)

				# memcpy fa
				for i in xrange(m):
					f_py[i] = lib.function(hval, a, b, c, d, e)

				lib.function_vector_memcpy_va(f, f_ptr)
				lib.function_vector_memcpy_av(f_ptr, f)

				f_list = [lib.function(*f_) for f_ in f_py]
				for i in xrange(m):
					self.assertAlmostEqual(f_list[i].h, hval)
					self.assertAlmostEqual(f_list[i].a, a, DIGITS)
					self.assertAlmostEqual(f_list[i].b, b, DIGITS)
					self.assertAlmostEqual(f_list[i].c, c, DIGITS)
					self.assertAlmostEqual(f_list[i].d, d, DIGITS)
					self.assertAlmostEqual(f_list[i].e, e, DIGITS)

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

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.prox_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if lib.FLOAT else 7

			f, f_py, f_ptr = self.make_prox_triplet(lib, m)
			self.register_var('f', f, lib.function_vector_free)

			v = dlib.vector(0, 0, None)
			dlib.vector_calloc(v, m)
			self.register_var('v', v, lib.vector_free)

			v_py = np.zeros(m).astype(dlib.pyfloat)
			v_ptr = v_py.ctypes.data_as(dlib.ok_float_p)

			for i in xrange(m):
				f_py[i] = lib.function(hval, a[i], b[i], c[i], d[i], e[i])
			lib.function_vector_memcpy_va(f, f_ptr)
			v_py[:] = np.random.rand(m)
			dlib.vector_memcpy_va(v, v_ptr, 1)

			# mul
			lib.function_vector_mul(f, v)
			lib.function_vector_memcpy_av(f_ptr, f)
			for i in xrange(m):
				a[i] *= v_py[i]
				d[i] *= v_py[i]
				e[i] *= v_py[i]
			f_list = [lib.function(*f_) for f_ in f_py]
			for i in xrange(m):
				self.assertAlmostEqual(f_list[i].h, hval, DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].a),
									   significant_digits(a[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].b),
									   significant_digits(b[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].c),
									   significant_digits(c[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].d),
									   significant_digits(d[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].e),
									   significant_digits(e[i]), DIGITS)

			# div
			lib.function_vector_div(f, v)
			lib.function_vector_memcpy_av(f_ptr, f)
			for i in xrange(m):
				a[i] /= v_py[i]
				d[i] /= v_py[i]
				e[i] /= v_py[i]
			f_list = [lib.function(*f_) for f_ in f_py]
			for i in xrange(m):
				self.assertAlmostEqual(f_list[i].h, hval, DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].a),
									   significant_digits(a[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].b),
									   significant_digits(b[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].c),
									   significant_digits(c[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].d),
									   significant_digits(d[i]), DIGITS)
				self.assertAlmostEqual(significant_digits(f_list[i].e),
									   significant_digits(e[i]), DIGITS)

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

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.prox_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 if lib.FLOAT else 7

			f, f_py, f_ptr = self.make_prox_triplet(lib, m)
			self.register_var('f', f, lib.function_vector_free)

			x = dlib.vector(0, 0, None)
			x_py = np.zeros(m).astype(dlib.pyfloat)
			x_ptr = x_py.ctypes.data_as(dlib.ok_float_p)
			dlib.vector_calloc(x, m)
			self.register_var('x', x, lib.vector_free)

			xout = dlib.vector(0, 0, None)
			xout_py = np.zeros(m).astype(dlib.pyfloat)
			xout_ptr = xout_py.ctypes.data_as(dlib.ok_float_p)
			dlib.vector_calloc(xout, m)
			self.register_var('xout', xout, lib.vector_free)

			# populate
			x_py += x_rand
			dlib.vector_memcpy_va(x, x_ptr, 1)

			for hkey, hval in lib.enums.dict.items():
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
					self.assertTrue(1)
				else:
					self.assertAlmostEqual(significant_digits(funcval_c),
										   significant_digits(funcval_py),
										   DIGITS)

				# proximal operator evaluation, random rho
				rho = 5 * np.random.rand()
				prox_py = prox_eval_python(f_list, rho, x_rand)
				lib.ProxEvalVector(f, rho, x, xout)
				dlib.vector_memcpy_av(xout_ptr, xout, 1)
				if not np.allclose(xout_py, prox_py, 2):
					self.assertTrue(np.allclose(xout_py, prox_py, 1))
				else:
					self.assertTrue(np.allclose(xout_py, prox_py, 2))

			self.free_var('f')
			self.free_var('x')
			self.free_var('xout')