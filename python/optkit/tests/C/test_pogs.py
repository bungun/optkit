import os
import numpy as np
from ctypes import c_void_p, byref, cast, addressof
from optkit.utils.proxutils import func_eval_python
from optkit.libs.pogs import PogsLibs
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
		self.assertTrue( any(libs) )

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

	def assert_pogs_equilibration(self, lib, solver_work, A, A_equil):
		m, n = A.shape

		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5

		d_local = np.zeros(m).astype(lib.pyfloat)
		e_local = np.zeros(n).astype(lib.pyfloat)
		self.load_to_local(lib, d_local, solver_work.contents.d)
		self.load_to_local(lib, e_local, solver_work.contents.e)

		x_rand = np.random.rand(n)
		A_eqx = A_equil.dot(x_rand)
		DAEx = d_local * A.dot(e_local * x_rand)
		self.assertVecEqual( A_eqx, DAEx, ATOLM, RTOL )

	def assert_pogs_projector(self, lib, blas_handle, projector, A_equil):
		m, n = A_equil.shape
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5

		x_in, x_in_py, x_in_ptr = self.register_vector(lib, n, 'x_in')
		y_in, y_in_py, y_in_ptr = self.register_vector(lib, m, 'y_in')
		x_out, x_out_py, x_out_ptr = self.register_vector(lib, n, 'x_out')
		y_out, y_out_py, y_out_ptr = self.register_vector(lib, m, 'y_out')

		x_in_py += np.random.rand(n)
		y_in_py += np.random.rand(m)
		self.assertCall( lib.vector_memcpy_va(x_in, x_in_ptr, 1) )
		self.assertCall( lib.vector_memcpy_va(y_in, y_in_ptr, 1) )

		if lib.direct:
			self.assertCall( lib.direct_projector_project(
					blas_handle, projector, x_in, y_in, x_out, y_out) )
		else:
			self.assertCall( lib.indirect_projector_project(
					blas_handle, projector, x_in, y_in, x_out, y_out) )

		self.load_to_local(lib, x_out_py, x_out)
		self.load_to_local(lib, y_out_py, y_out)

		self.assertVecEqual(
				A_equil.dot(x_out_py), y_out_py, ATOLM, RTOL )

		self.free_vars('x_in', 'y_in', 'x_out', 'y_out')

	def assert_pogs_primal_project(self, lib, blas_handle, solver, localA,
								   local_vars):
		"""primal projection test
			set
				(x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})
			check that
				y^{k+1} == A * x^{k+1}
			holds to numerical tolerance
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * local_vars.m**0.5

		projector = solver.contents.M.contents.P

		self.assertCall( lib.project_primal(blas_handle, projector,
				solver.contents.z, solver.contents.settings.contents.alpha) )
		self.load_all_local(lib, local_vars, solver)
		self.assertVecEqual( localA.dot(local_vars.x), local_vars.y, ATOLM, RTOL)

	def assert_pogs_warmstart(self, lib, solver, A_equil, local_vars, x0, nu0):
		m, n = A_equil.shape
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

		rho = solver.contents.rho
		self.load_all_local(lib, local_vars, solver)

		# check variable state is consistent after warm start
		self.assertVecEqual(
				x0, local_vars.e * local_vars.x, ATOLN, RTOL )
		self.assertVecEqual(
				nu0 * -1./rho, local_vars.d * local_vars.yt,
				ATOLM, RTOL )
		self.assertVecEqual(
				A_equil.dot(x0 / local_vars.e), local_vars.y, ATOLM, RTOL )
		self.assertVecEqual(
				A_equil.T.dot(nu0 / (rho * local_vars.d)), local_vars.xt,
				ATOLN, RTOL )

	def assert_pogs_check_convergence(self, lib, blas_handle, solver, f_list,
									  g_list, objectives, residuals, tolerances,
									  localA, local_vars):
		"""convergence test

			(1) set

				obj_primal = f(y^{k+1/2}) + g(x^{k+1/2})
				obj_gap = <z^{k+1/2}, zt^{k+1/2}>
				obj_dual = obj_primal - obj_gap

				tol_primal = abstol * sqrt(m) + reltol * ||y^{k+1/2}||
				tol_dual = abstol * sqrt(n) + reltol * ||xt^{k+1/2}||

				res_primal = ||Ax^{k+1/2} - y^{k+1/2}||
				res_dual = ||A'yt^{k+1/2} + xt^{k+1/2}||

			in C and Python, check that these quantities agree

			(2) calculate solver convergence,

					res_primal <= tol_primal
					res_dual <= tol_dual,

				in C and Python, check that the results agree
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)

		converged = lib.check_convergence(blas_handle, solver, objectives,
										  residuals, tolerances)

		self.load_all_local(lib, local_vars, solver)
		obj_py = func_eval_python(g_list, local_vars.x12)
		obj_py += func_eval_python(f_list, local_vars.y12)
		obj_gap_py = abs(local_vars.z12.dot(local_vars.zt12))
		obj_dua_py = obj_py - obj_gap_py

		tol_primal = tolerances.atolm + (
				tolerances.reltol * np.linalg.norm(local_vars.y12))
		tol_dual = tolerances.atoln + (
				tolerances.reltol * np.linalg.norm(local_vars.xt12))

		self.assertScalarEqual( objectives.gap, obj_gap_py, RTOL )
		self.assertScalarEqual( tolerances.primal, tol_primal, RTOL )
		self.assertScalarEqual( tolerances.dual, tol_dual, RTOL )

		res_primal = np.linalg.norm(
				localA.dot(local_vars.x12) - local_vars.y12)
		res_dual = np.linalg.norm(
				localA.T.dot(local_vars.yt12) + local_vars.xt12)

		self.assertScalarEqual( residuals.primal, res_primal, RTOL )
		self.assertScalarEqual( residuals.dual, res_dual, RTOL )
		self.assertScalarEqual( residuals.gap, abs(obj_gap_py), RTOL )

		converged_py = res_primal <= tolerances.primal and \
					   res_dual <= tolerances.dual

		self.assertEqual( converged, converged_py )

	def test_default_settings(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.assert_default_settings(lib)

	def test_pogs_init_finish(self, reset=0):
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

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
			elif not lib.full_api_accessible:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')
				f, f_py, g, g_py = self.gen_registered_pogs_fns(lib, m, n)
				f_list = [lib.function(*f_) for f_ in f_py]
				g_list = [lib.function(*g_) for g_ in g_py]

				# problem matrix A
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				solver = lib.pogs_init(A_ptr, m, n, order)
				self.register_solver('solver', solver, lib.pogs_finish)
				output, info, settings = self.gen_pogs_params(lib, m, n)

				local_vars = self.PogsVariablesLocal(m, n, lib.pyfloat)
				localA, localA_ptr = self.gen_py_matrix(lib, m, n, order)

				self.assertCall( lib.matrix_memcpy_am(
						localA_ptr, solver.contents.M.contents.A, order) )

				res = lib.pogs_residuals()
				tols = lib.pogs_tolerances()
				obj = lib.pogs_objectives()
				self.assertCall( lib.initialize_conditions(
						obj, res, tols, settings, m, n) )

				# test (coldstart) solver calls
				z = solver.contents.z
				M = solver.contents.M
				rho = solver.contents.rho
				self.assert_pogs_equilibration(lib, M, A, localA)
				self.assert_pogs_projector(lib, hdl, M.contents.P, localA)
				self.assert_pogs_scaling(lib, solver, f, f_py, g, g_py,
										 local_vars)
				self.assert_pogs_primal_update(lib, solver, local_vars)
				self.assert_pogs_prox(lib, hdl, solver, f, f_py, g, g_py,
									  local_vars)
				self.assert_pogs_primal_project(lib, hdl, solver, localA,
												local_vars)
				self.assert_pogs_dual_update(lib, hdl, solver, local_vars)
				self.assert_pogs_check_convergence(lib, hdl, solver, f_list,
												   g_list, obj, res, tols,
												   localA, local_vars)
				self.assert_pogs_adapt_rho(lib, solver, res, tols, local_vars)
				self.assert_pogs_unscaling(lib, output, solver, local_vars)

				x0, x0_ptr = self.gen_py_vector(lib, n, random=True)
				nu0, nu0_ptr = self.gen_py_vector(lib, m, random=True)

				settings.x0 = x0_ptr
				settings.nu0 = nu0_ptr
				self.assertCall( lib.update_settings(solver.contents.settings,
													 byref(settings)) )
				self.assertCall( lib.initialize_variables(solver) )

				self.assert_pogs_warmstart(lib, solver, localA, local_vars, x0,
										   nu0)

				self.free_vars('solver', 'f', 'g', 'hdl')
				self.assertCall( lib.ok_device_reset() )

	def test_pogs_call(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				f, f_py, g, g_py = self.gen_registered_pogs_fns(lib, m, n)

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
					self.assert_pogs_convergence(
							A, settings, output, gpu=gpu,
							single_precision=single_precision)

				self.free_vars('f', 'g')
				self.assertCall( lib.ok_device_reset() )

	def test_pogs_call_unified(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				f, f_py, g, g_py = self.gen_registered_pogs_fns(lib, m, n)

				# problem matrix
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				# i/o
				output, info, settings = self.gen_pogs_params(lib, m, n)
				settings.verbose = 1

				# pogs
				self.assertCall( lib.pogs(A_ptr, f, g, settings, info,
										  output.ptr, order, 0) )

				if info.converged:
					self.assert_pogs_convergence(
							A, settings, output, gpu=gpu,
							single_precision=single_precision)

				self.free_vars('f', 'g')
				self.assertCall( lib.ok_device_reset() )

	def test_pogs_warmstart(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * lib.FLOAT
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5
			ATOLN = RTOL * n**0.5

			x0, x0_ptr = self.gen_py_vector(lib, n, random=True)
			nu0, nu0_ptr = self.gen_py_vector(lib, m, random=True)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				f, f_py, g, g_py = self.gen_registered_pogs_fns(lib, m, n)

				# problem matrix
				A, A_ptr = self.gen_py_matrix(lib, m, n, order)
				A += self.A_test

				# solver
				solver = lib.pogs_init(A_ptr, m, n, order)
				self.register_solver('solver', solver, lib.pogs_finish)
				output, info, settings = self.gen_pogs_params(lib, m, n)

				# warm start settings
				settings.x0 = x0_ptr
				settings.nu0 = nu0_ptr
				settings.warmstart = 1
				settings.maxiter = 0

				if self.VERBOSE_TEST:
					print "\nwarm start variable loading test (0 iters)"
				self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
												output.ptr) )
				self.assertEqual( info.err, 0 )
				self.assertTrue( info.converged or info.k >= settings.maxiter )

				# CHECK VARIABLE INPUT
				if lib.full_api_accessible:
					local_vars = self.PogsVariablesLocal(m, n, lib.pyfloat)
					localA, localA_ptr = self.gen_py_matrix(lib, m, n, order)

					self.assertCall( lib.matrix_memcpy_am(
							localA_ptr, solver.contents.M.contents.A, order) )

					self.assert_pogs_warmstart(
							lib, solver, localA, local_vars, x0, nu0)

				# WARMSTART SOLVE SEQUENCE
				self.assert_warmstart_sequence(lib, solver, f, g, settings,
											   info, output)

				self.free_vars('solver', 'f', 'g')
				self.assertCall( lib.ok_device_reset() )

	def test_pogs_io(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			x_rand, _ = self.gen_py_vector(lib, n)
			nu_rand, _ = self.gen_py_vector(lib, m)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				f, f_py, g, g_py = self.gen_registered_pogs_fns(lib, m, n)

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
				self.free_vars('solver', 'f', 'g')
				self.assertCall( lib.ok_device_reset() )