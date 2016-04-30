import os
import numpy as np
from ctypes import c_void_p, byref, cast, addressof
from optkit.libs import PogsLibs
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.pogs_base import OptkitCPogsTestCase

class PogsLibsTestCase(OptkitTestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = PogsLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue(any(libs))

class PogsTestCase(OptkitCPogsTestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = PogsLibs()
		self.A_test = self.A_test_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_default_settings(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.assert_default_settings(lib)

	def test_pogs_init_finish(self, reset=0):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test
				solver = lib.pogs_init(A_ptr, m, n, order)
				self.assertCall( lib.pogs_finish(solver, 1) )

	def test_pogs_private_api(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			if not lib.full_api_accessible:
				continue

			hdl = self.register_blas_handle(lib, 'hdl')
			f, f_py, g, g_py = self.gen_registered_pogs_test_vars(lib, m, n)
			f_list = [lib.function(*f_) for f_ in f_py]
			g_list = [lib.function(*g_) for g_ in g_py]

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):

				# problem matrix A
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				solver = lib.pogs_init(A_ptr, m, n, order)
				self.register_solver('solver', solver, lib.pogs_finish)
				output, info, settings = self.gen_pogs_params(lib, m, n)


				localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)
				localA, localA_ptr = self.gen_py_matrix(lib, m, n, order)

				self.assertCall( lib.matrix_memcpy_am(
						localA_ptr, solver.contents.M.contents.A, order) )

				res = lib.pogs_residuals(0, 0, 0)
				tols = lib.make_tolerances(settings, m, n)
				obj = lib.pogs_objectives(0, 0, 0)

				# test (coldstart) solver calls
				self.assert_pogs_equilibration(lib, solver, A, localA, localvars)
				self.assert_pogs_projector(lib, hdl, solver, localA)
				self.assert_pogs_scaling(lib, solver, f, f_py, g, g_py, localvars)
				self.assert_pogs_primal_update(lib, solver, localvars)
				self.assert_pogs_prox(lib, hdl, solver, f, f_py, g, g_py, localvars)
				self.assert_pogs_primal_project(lib, hdl, solver, settings, localA,
										 localvars)
				self.assert_pogs_dual_update(lib, hdl, solver, settings,localvars)
				self.assert_pogs_check_convergence(lib, hdl, solver, f_list, g_list,
											obj, res, tols, settings, localA,
											localvars)
				self.assert_pogs_adapt_rho(lib, solver, settings, res, tols,
										   localvars)
				self.assert_pogs_unscaling(lib, solver, output, localvars)

				# test (warmstart) variable initialization:
				settings.x0 = x_rand.ctypes.data_as(lib.ok_float_p)
				settings.nu0 = nu_rand.ctypes.data_as(lib.ok_float_p)
				self.assertCall( lib.update_settings(solver.contents.settings,
													 settings) )
				self.assertCall( lib.initialize_variables(solver) )
				self.assert_pogs_warmstart(lib, solver.contents.rho,
										   solver.contents.z,
										   solver.contents.M, settings, localA,
										   localvars)

				self.free_var('solver')

			self.free_vars('f', 'g', 'hdl')
			self.assertCall( lib.ok_device_reset() )

	def test_pogs_call(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			f, f_py, g, g_py = self.gen_registered_pogs_test_vars(lib, m, n)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):

				# problem matrix
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				solver = lib.pogs_init(A_ptr, m, n, order)
				self.register_solver('solver', solver, lib.pogs_finish)
				output, info, settings = self.gen_pogs_params(lib, m, n)

				# solve
				self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
												output.ptr) )
				self.free_var('solver')

				if info.converged:
					self.assert_pogs_convergence(A, settings, output)

			self.free_vars('f', 'g')
			self.assertCall( lib.ok_device_reset() )

	def test_pogs_call_unified(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			f, f_py, g, g_py = self.gen_registered_pogs_test_vars(lib, m, n)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):

				# problem matrix
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				# i/o
				output, info, settings = self.gen_pogs_params(lib, m, n)

				# pogs
				self.assertCall( lib.pogs(A_ptr, f, g, settings, info,
										  output.ptr, order, 0) )

				if info.converged:
					self.assert_pogs_convergence(A, settings, output)

			self.free_vars('f', 'g')
			self.assertCall( lib.ok_device_reset() )

	def test_pogs_warmstart(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			x_rand, _ = self.gen_py_vector(lib, n)
			nu_rand, _ = self.gen_py_vector(lib, m)


			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):


				f, f_py, g, g_py = self.gen_registered_pogs_test_vars(lib, m, n)

				# problem matrix
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				# solver
				solver = lib.pogs_init(A_ptr, m, n, order)
				self.register_solver('solver', solver, lib.pogs_finish)
				output, info, settings = self.gen_pogs_params(lib, m, n)

				# warm start settings
				settings.x0 = x_rand.ctypes.data_as(lib.ok_float_p)
				settings.nu0 = nu_rand.ctypes.data_as(lib.ok_float_p)
				settings.warmstart = 1
				settings.maxiter = 0

				print "\nwarm start variable loading test (0 iters)"
				self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
												output.ptr) )
				self.assertEqual( info.err, 0 )
				self.assertTrue( info.converged or info.k >= settings.maxiter )

				# CHECK VARIABLE INPUT
				localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)
				self.assert_pogs_warmstart(
						lib, solver.contents.rho, solver.contents.z,
						solver.contents.M, settings, localA, localvars)

				# WARMSTART SOLVE SEQUENCE
				self.assert_warmstart_sequence(lib, solver, f, g, settings,
											   info, output)
				self.free_var('solver', 'f', 'g')

			self.assertCall( lib.ok_device_reset() )

			self.free_vars('f', 'g')

	def test_pogs_io(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			f, f_py, g, g_py = self.gen_registered_pogs_test_vars(lib, m, n)
			x_rand, _ = self.gen_py_vector(lib, n)
			nu_rand, _ = self.gen_py_vector(lib, m)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):

				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor

				# problem matrix
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				# solver
				solver = lib.pogs_init(A_ptr, m, n, order)
				self.register_solver('solver', solver, lib.pogs_finish)
				output, info, settings = self.gen_pogs_params(lib, m, n)
				settings.verbose = 1

				# solve
				print 'initial solve -> export data'
				self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
								 output.ptr) )
				k_orig = info.k

				# placeholders for problem state
				A_equil, A_equil_ptr = self.gen_py_matrix(lib, m, n, order)
				if lib.direct:
					k = min(m, n)
					LLT, LLT_ptr = self.gen_py_matrix(lib, k, k, order)
				else:
					LLT_ptr = LLT = c_void_p()

				d, d_ptr = self.gen_py_vector(lib, m)
				e, e_ptr = self.gen_py_vector(lib, n)
				z, z_ptr = self.gen_py_vector(lib, m + n)
				z12, z12_ptr = self.gen_py_vector(lib, m + n)
				zt, zt_ptr = self.gen_py_vector(lib, m + n)
				zt12, zt12_ptr = self.gen_py_vector(lib, m + n)
				zprev, zprev_ptr = self.gen_py_vector(lib, m + n)
				rho, rho_ptr = self.gen_py_vector(lib, 1)

				# copy state out
				self.assertCall( lib.pogs_extract_solver(
						solver, A_equil_ptr, LLT_ptr, d_ptr, e_ptr, z_ptr,
						z12_ptr, zt_ptr, zt12_ptr, zprev_ptr, rho_ptr, order) )
				self.free_var('solver')

				# copy state in to new solver
				solver = lib.pogs_load_solver(
						A_equil_ptr, LLT_ptr, d_ptr, e_ptr, z_ptr, z12_ptr,
						zt_ptr, zt12_ptr, zprev_ptr, rho[0], m, n, order)
				self.register_solver('solver', solver, lib.pogs_finish)

				settings.resume = 1
				print 'import data -> solve again'
				self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
												output.ptr) )
				self.assertTrue(info.k <= k_orig or not info.converged)
				self.free_var('solver')

			self.free_vars('f', 'g')
			self.assertEqual(lib.ok_device_reset(), 0)