import os
import numpy as np
from ctypes import c_int, byref, c_void_p
from optkit.libs.prox import ProxLibs
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.defs import OptkitTestCase
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
		self.assertTrue( any(libs) )

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
			if self.VERBOSE_TEST:
				print("proxlib version", version)

class ProxTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProxLibs()
		self.scalefactor = 5

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		pass

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_alloc(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			# calloc
			f = lib.function_vector(0, None)
			self.assertEqual( f.size, 0 )

			self.assertCall( lib.function_vector_calloc(f, m) )
			self.register_var('f', f, lib.function_vector_free)
			self.assertEqual( f.size, m )

			# free
			self.free_var('f')
			self.assertEqual( f.size, 0 )

			self.assertCall( lib.ok_device_reset() )

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
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			f, f_py, f_ptr = self.register_fnvector(lib, m, 'f')

			# initialize to default values
			hlast = 0
			alast = 1.
			blast = 0.
			clast = 1.
			dlast = 0.
			elast = 0.

			for hkey, hval in lib.function_enums.dict.items():
				if self.VERBOSE_TEST:
					print hkey

				for i in xrange(m):
					f_py[i] = lib.function(hval, a, b, c, d, e)

				f_list = [lib.function(*f_) for f_ in f_py]
				fh = np.array([f_.h - hval for f_ in f_list])
				fa = np.array([f_.a - a for f_ in f_list])
				fb = np.array([f_.b - b for f_ in f_list])
				fc = np.array([f_.c - c for f_ in f_list])
				fd = np.array([f_.d - d for f_ in f_list])
				fe = np.array([f_.e - e for f_ in f_list])
				self.assertVecEqual( 0, fh, ATOLM, 0 )
				self.assertVecEqual( 0, fa, ATOLM, 0 )
				self.assertVecEqual( 0, fb, ATOLM, 0 )
				self.assertVecEqual( 0, fc, ATOLM, 0 )
				self.assertVecEqual( 0, fd, ATOLM, 0 )
				self.assertVecEqual( 0, fe, ATOLM, 0 )

				# memcpy af
				self.assertCall( lib.function_vector_memcpy_av(f_ptr, f) )
				f_list = [lib.function(*f_) for f_ in f_py]

				fh = np.array([f_.h - hlast for f_ in f_list])
				fa = np.array([f_.a - alast for f_ in f_list])
				fb = np.array([f_.b - blast for f_ in f_list])
				fc = np.array([f_.c - clast for f_ in f_list])
				fd = np.array([f_.d - dlast for f_ in f_list])
				fe = np.array([f_.e - elast for f_ in f_list])
				self.assertVecEqual( 0, fh, ATOLM, 0 )
				self.assertVecEqual( 0, fa, ATOLM, 0 )
				self.assertVecEqual( 0, fb, ATOLM, 0 )
				self.assertVecEqual( 0, fc, ATOLM, 0 )
				self.assertVecEqual( 0, fd, ATOLM, 0 )
				self.assertVecEqual( 0, fe, ATOLM, 0 )

				# memcpy fa
				for i in xrange(m):
					f_py[i] = lib.function(hval, a, b, c, d, e)

				self.assertCall( lib.function_vector_memcpy_va(f, f_ptr) )
				self.assertCall( lib.function_vector_memcpy_av(f_ptr, f) )

				f_list = [lib.function(*f_) for f_ in f_py]
				fh = np.array([f_.h - hval for f_ in f_list])
				fa = np.array([f_.a - a for f_ in f_list])
				fb = np.array([f_.b - b for f_ in f_list])
				fc = np.array([f_.c - c for f_ in f_list])
				fd = np.array([f_.d - d for f_ in f_list])
				fe = np.array([f_.e - e for f_ in f_list])
				self.assertVecEqual( 0, fh, ATOLM, 0 )
				self.assertVecEqual( 0, fa, ATOLM, 0 )
				self.assertVecEqual( 0, fb, ATOLM, 0 )
				self.assertVecEqual( 0, fc, ATOLM, 0 )
				self.assertVecEqual( 0, fd, ATOLM, 0 )
				self.assertVecEqual( 0, fe, ATOLM, 0 )

				hlast = hval
				alast = a
				blast = b
				clast = c
				dlast = d
				elast = e

			self.free_var('f')
			self.assertCall( lib.ok_device_reset() )

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
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			f, f_py, f_ptr = self.register_fnvector(lib, m, 'f')
			self.register_var('f', f, lib.function_vector_free)

			v = lib.vector(0, 0, None)
			self.assertCall( lib.vector_calloc(v, m) )
			self.register_var('v', v, lib.vector_free)

			v_py = np.zeros(m).astype(lib.pyfloat)
			v_ptr = v_py.ctypes.data_as(lib.ok_float_p)

			for i in xrange(m):
				f_py[i] = lib.function(hval, a[i], b[i], c[i], d[i], e[i])
			self.assertCall( lib.function_vector_memcpy_va(f, f_ptr) )
			v_py[:] = np.random.rand(m)
			self.assertCall( lib.vector_memcpy_va(v, v_ptr, 1) )

			# mul
			self.assertCall( lib.function_vector_mul(f, v) )
			self.assertCall( lib.function_vector_memcpy_av(f_ptr, f) )
			for i in xrange(m):
				a[i] *= v_py[i]
				d[i] *= v_py[i]
				e[i] *= v_py[i]
			f_list = [lib.function(*f_) for f_ in f_py]
			fh = np.array([f_.h - hval for f_ in f_list])
			fa = np.array([f_.a for f_ in f_list])
			fb = np.array([f_.b for f_ in f_list])
			fc = np.array([f_.c for f_ in f_list])
			fd = np.array([f_.d for f_ in f_list])
			fe = np.array([f_.e for f_ in f_list])
			self.assertVecEqual( 0, fh, ATOLM, 0 )
			self.assertVecEqual( fa, a , ATOLM, RTOL )
			self.assertVecEqual( fb, b , ATOLM, RTOL )
			self.assertVecEqual( fc, c , ATOLM, RTOL )
			self.assertVecEqual( fd, d , ATOLM, RTOL )
			self.assertVecEqual( fe, e , ATOLM, RTOL )
			# div
			self.assertCall( lib.function_vector_div(f, v) )
			self.assertCall( lib.function_vector_memcpy_av(f_ptr, f) )
			for i in xrange(m):
				a[i] /= v_py[i]
				d[i] /= v_py[i]
				e[i] /= v_py[i]
			f_list = [lib.function(*f_) for f_ in f_py]
			fh = np.array([f_.h - hval for f_ in f_list])
			fa = np.array([f_.a for f_ in f_list])
			fb = np.array([f_.b for f_ in f_list])
			fc = np.array([f_.c for f_ in f_list])
			fd = np.array([f_.d for f_ in f_list])
			fe = np.array([f_.e for f_ in f_list])
			self.assertVecEqual( 0, fh, ATOLM, 0)
			self.assertVecEqual( fa, a , ATOLM, RTOL )
			self.assertVecEqual( fb, b , ATOLM, RTOL )
			self.assertVecEqual( fc, c , ATOLM, RTOL )
			self.assertVecEqual( fd, d , ATOLM, RTOL )
			self.assertVecEqual( fe, e , ATOLM, RTOL )

			self.free_vars('f', 'v')
			self.assertCall( lib.ok_device_reset() )

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
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			f, f_py, f_ptr = self.register_fnvector(lib, m, 'f')
			x, x_py, x_ptr = self.register_vector(lib, m, 'x')
			xout, xout_py, xout_ptr = self.register_vector(lib, m, 'xout')

			# populate
			x_py += x_rand
			self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

			for hkey, hval in lib.function_enums.dict.items():
				if self.VERBOSE_TEST:
					print hkey

				print hkey
				# avoid domain errors with randomly generated data
				if 'Log' in hkey or 'Exp' in hkey or 'Entr' in hkey:
					continue

				for i in xrange(m):
					f_py[i] = lib.function(hval, a[i], b[i], c[i], d[i], e[i])
				self.assertCall( lib.function_vector_memcpy_va(f, f_ptr) )

				# function evaluation
				f_list = [lib.function(*f_) for f_ in f_py]
				funcval_py = func_eval_python(f_list, x_rand)

				funcval_c = np.zeros(1).astype(lib.pyfloat)
				funcval_c_ptr = funcval_c.ctypes.data_as(lib.ok_float_p)
				self.assertCall( lib.function_eval_vector(f, x,
														  funcval_c_ptr) )

				if funcval_c[0] in (np.inf, np.nan):
					self.assertTrue( 1 )
				else:
					self.assertScalarEqual( funcval_py, funcval_c, RTOL )

				# proximal operator evaluation, random rho
				rho = 5 * np.random.rand()
				prox_py = prox_eval_python(f_list, rho, x_rand)
				self.assertCall( lib.prox_eval_vector(f, rho, x, xout) )
				self.assertCall( lib.vector_memcpy_av(xout_ptr, xout, 1) )
				self.assertVecEqual( xout_py, prox_py, ATOLM, RTOL )

			self.free_vars('f', 'x', 'xout')
			self.assertCall( lib.ok_device_reset() )