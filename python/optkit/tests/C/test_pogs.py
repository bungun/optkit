import os
import numpy as np
from ctypes import c_void_p, byref, cast, addressof
from optkit.libs import PogsLibs
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.base import OptkitCPogsTestCase

ALPHA_DEFAULT = 1.7
RHO_DEFAULT = 1
MAXITER_DEFAULT = 2000
ABSTOL_DEFAULT = 1e-4
RELTOL_DEFAULT = 1e-3
ADAPTIVE_DEFAULT = 1
GAPSTOP_DEFAULT = 0
WARMSTART_DEFAULT = 0
VERBOSE_DEFAULT = 2
SUPPRESS_DEFAULT = 0
RESUME_DEFAULT = 0

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

	@staticmethod
	def load_to_local(lib, py_vector, c_vector):
		lib.vector_memcpy_av(
				py_vector.ctypes.data_as(lib.ok_float_p), c_vector, 1)

	@staticmethod
	def load_all_local(lib, py_vars, solver):
		if not isinstance(py_vars, self.PogsVariablesLocal):
			raise TypeError('argument "py_vars" must be of type {}'.format(
							self.PogsVariablesLocal))

		self.load_to_local(lib, py_vars.z,
						   solver.contents.z.contents.primal.contents.vec)
		self.load_to_local(lib, py_vars.z12,
						   solver.contents.z.contents.primal12.contents.vec)
		self.load_to_local(lib, py_vars.zt,
						   solver.contents.z.contents.dual.contents.vec)
		self.load_to_local(lib, py_vars.zt12,
						   solver.contents.z.contents.dual12.contents.vec)
		self.load_to_local(lib, py_vars.prev,
						   solver.contents.z.contents.prev.contents.vec)
		self.load_to_local(lib, py_vars.d, solver.contents.M.contents.d)
		self.load_to_local(lib, py_vars.e, solver.contents.M.contents.e)


	def pogs_equilibration(self, lib, solver, A, localA, localvars):
		DIGITS = 7 - 2 * lib.FLOAT

		m, n = A.shape
		d_local = np.zeros(m).astype(lib.pyfloat)
		e_local = np.zeros(n).astype(lib.pyfloat)
		self.load_to_local(lib, d_local, solver.contents.M.contents.d)
		self.load_to_local(lib, e_local, solver.contents.M.contents.e)

		x_rand = np.random.rand(n)
		A_eqx = localA.dot(x_rand)
		DAEx = d_local * A.dot(e_local * x_rand)
		self.assertTrue(np.allclose(A_eqx, DAEx, DIGITS))

	def pogs_projector(self, lib, blas_handle, solver, localA):
		DIGITS = 7 - 2 * lib.FLOAT

		m, n = localA.shape

		x_in = lib.vector(0, 0, None)
		y_in = lib.vector(0, 0, None)
		x_out = lib.vector(0, 0, None)
		y_out = lib.vector(0, 0, None)

		lib.vector_calloc(x_in, n)
		lib.vector_calloc(y_in, m)
		lib.vector_calloc(x_out, n)
		lib.vector_calloc(y_out, m)
		self.register_var('x_in', x_in, lib.vector_free)
		self.register_var('y_in', y_in, lib.vector_free)
		self.register_var('x_out', x_out, lib.vector_free)
		self.register_var('y_out', y_out, lib.vector_free)

		x_in_py = np.random.rand(n).astype(lib.pyfloat)
		y_in_py = np.random.rand(m).astype(lib.pyfloat)
		x_out_py = np.zeros(n).astype(lib.pyfloat)
		y_out_py = np.zeros(m).astype(lib.pyfloat)

		x_in_ptr = x_in_py.ctypes.data_as(lib.ok_float_p)
		y_in_ptr = y_in_py.ctypes.data_as(lib.ok_float_p)

		lib.vector_memcpy_va(x_in, x_in_ptr, 1)
		lib.vector_memcpy_va(y_in, y_in_ptr, 1)


		if lib.direct:
			lib.direct_projector_project(blas_handle,
											 solver.contents.M.contents.P,
											 x_in, y_in, x_out, y_out)
		else:
			lib.indirect_projector_project(blas_handle,
											 solver.contents.M.contents.P,
											 x_in, y_in, x_out, y_out)

		self.load_to_local(lib, x_out_py, x_out)
		self.load_to_local(lib, y_out_py, y_out)

		self.assertTrue(np.allclose(localA.dot(x_out_py), y_out_py, DIGITS))

		self.free_var('x_in')
		self.free_var('y_in')
		self.free_var('x_out')
		self.free_var('y_out')

	def pogs_scaling(self, lib, solver, f, f_py, g, g_py, localvars):
		DIGITS = 7 - 2 * lib.FLOAT

		def fv_list2arrays(function_vector_list):
			fv = function_vector_list
			fvh = [fv_.h for fv_ in fv]
			fva = [fv_.a for fv_ in fv]
			fvb = [fv_.b for fv_ in fv]
			fvc = [fv_.c for fv_ in fv]
			fvd = [fv_.d for fv_ in fv]
			fve = [fv_.e for fv_ in fv]
			return fvh, fva, fvb, fvc, fvd, fve

		# record original function vector parameters
		f_list = [lib.function(*f_) for f_ in f_py]
		g_list = [lib.function(*f_) for f_ in g_py]
		f_h0, f_a0, f_b0, f_c0, f_d0, f_e0 = fv_list2arrays(f_list)
		g_h0, g_a0, g_b0, g_c0, g_d0, g_e0 = fv_list2arrays(g_list)

		# copy function vector
		f_py_ptr = f_py.ctypes.data_as(lib.function_p)
		g_py_ptr = g_py.ctypes.data_as(lib.function_p)
		lib.function_vector_memcpy_va(f, f_py_ptr)
		lib.function_vector_memcpy_va(g, g_py_ptr)

		# scale function vector
		lib.update_problem(solver, f, g)

		# retrieve scaled function vector parameters
		lib.function_vector_memcpy_av(f_py_ptr, solver.contents.f)
		lib.function_vector_memcpy_av(g_py_ptr, solver.contents.g)
		f_list = [lib.function(*f_) for f_ in f_py]
		g_list = [lib.function(*f_) for f_ in g_py]
		f_h1, f_a1, f_b1, f_c1, f_d1, f_e1 = fv_list2arrays(f_list)
		g_h1, g_a1, g_b1, g_c1, g_d1, g_e1 = fv_list2arrays(g_list)


		# retrieve scaling
		self.load_all_local(lib, localvars, solver)

		# scaled vars
		self.assertTrue(np.allclose(f_a0, localvars.d * f_a1, DIGITS))
		self.assertTrue(np.allclose(f_d0, localvars.d * f_d1, DIGITS))
		self.assertTrue(np.allclose(f_e0, localvars.d * f_e1, DIGITS))
		self.assertTrue(np.allclose(g_a0 * localvars.e, g_a1, DIGITS))
		self.assertTrue(np.allclose(g_d0 * localvars.e, g_d1, DIGITS))
		self.assertTrue(np.allclose(g_e0 * localvars.e, g_e1, DIGITS))

		# unchanged vars
		self.assertTrue(np.allclose(f_h0, f_h1, DIGITS))
		self.assertTrue(np.allclose(f_b0, f_b1, DIGITS))
		self.assertTrue(np.allclose(f_c0, f_c1, DIGITS))
		self.assertTrue(np.allclose(g_h0, g_h1, DIGITS))
		self.assertTrue(np.allclose(g_b0, g_b1, DIGITS))
		self.assertTrue(np.allclose(g_c0, g_c1, DIGITS))

	def pogs_warmstart(self, lib, solver, settings, localA,
					   localvars):

		DIGITS = 7 - 2 * lib.FLOAT

		m, n = localA.shape

		rho = solver.contents.rho

		x_rand = np.random.rand(n).astype(lib.pyfloat)
		nu_rand = np.random.rand(m).astype(lib.pyfloat)

		x_ptr = x_rand.ctypes.data_as(lib.ok_float_p)
		nu_ptr = nu_rand.ctypes.data_as(lib.ok_float_p)

		settings.x0 = x_ptr
		settings.nu0 = nu_ptr
		lib.update_settings(solver.contents.settings, settings)

		lib.initialize_variables(solver)
		self.load_all_local(lib, localvars, solver)

		self.assertTrue(np.allclose(x_rand, localvars.e * localvars.x, DIGITS))
		self.assertTrue(np.allclose(nu_rand * localvars.d * -1/rho,
									localvars.yt, DIGITS))

		self.assertTrue(np.allclose(localA.dot(localvars.x), localvars.y,
									DIGITS))
		self.assertTrue(np.allclose(localA.T.dot(localvars.yt), -localvars.xt,
									DIGITS))


	def pogs_primal_update(self, lib, solver, localvars):
		"""primal update test

			set

				z^k = z^{k-1}

			check

				z^k == z^{k-1}

			holds elementwise
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		lib.set_prev(solver.contents.z)
		self.load_all_local(lib, localvars, solver)
		self.assertTrue(np.allclose(localvars.z, localvars.prev, DIGITS))


	def pogs_prox(self, lib, blas_handle, solver, f, f_py, g, g_py, localvars):
		"""proximal operator application test

			set

				x^{k+1/2} = prox_{g, rho}(x^k - xt^k)
				y^{k+1/2} = prox_{f, rho}(y^k - yt^k)

			in C and Python, check that results agree
		"""
		DIGITS = 7 - 2 * lib.FLOAT

		lib.prox(blas_handle, f, g, solver.contents.z,
						 solver.contents.rho)
		self.load_all_local(lib, localvars, solver)

		f_list = [lib.function(*f_) for f_ in f_py]
		g_list = [lib.function(*f_) for f_ in g_py]
		for i in xrange(len(f_py)):
			f_list[i].a *= localvars.d[i]
			f_list[i].d *= localvars.d[i]
			f_list[i].e *= localvars.d[i]
		for j in xrange(len(g_py)):
			g_list[j].a /= localvars.e[j]
			g_list[j].d /= localvars.e[j]
			g_list[j].e /= localvars.e[j]

		x_arg = localvars.x - localvars.xt
		y_arg = localvars.y - localvars.yt
		x_out = prox_eval_python(g_list, solver.contents.rho, x_arg)
		y_out = prox_eval_python(f_list, solver.contents.rho, y_arg)
		self.assertTrue(np.allclose(localvars.x12, x_out, DIGITS))
		self.assertTrue(np.allclose(localvars.y12, y_out, DIGITS))

	def pogs_primal_project(self, lib, blas_handle, solver,
							settings, localA, localvars):
		"""primal projection test

			set

				(x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})

			check that


				y^{k+1/2} == A * x^{k+1/2}

			holds to numerical tolerance
		"""
		DIGITS = 7 - 2 * lib.FLOAT

		lib.project_primal(blas_handle, solver.contents.M.contents.P,
							   solver.contents.z, settings.alpha)
		self.load_all_local(lib, localvars, solver)
		self.assertTrue(np.allclose(localA.dot(localvars.x), localvars.y,
									DIGITS))

	def pogs_dual_update(self, lib, blas_handle, solver,
						 settings, localvars):
		"""dual update test

			set

				zt^{k+1/2} = z^{k+1/2} - z^{k} + zt^{k}
				zt^{k+1} = zt^{k} - z^{k+1} + alpha * z^{k+1/2} +
											  (1-alpha) * z^k

			in C and Python, check that results agree
		"""
		DIGITS = 7 - 2 * lib.FLOAT

		self.load_all_local(lib, localvars, solver)
		alpha = settings.alpha
		zt12_py = localvars.z12 - localvars.prev + localvars.zt
		zt_py = localvars.zt - localvars.z + (alpha * localvars.z12 +
								 			   (1-alpha) * localvars.prev)

		lib.update_dual(blas_handle, solver.contents.z, alpha)
		self.load_all_local(lib, localvars, solver)
		self.assertTrue(np.allclose(localvars.zt12, zt12_py, DIGITS))
		self.assertTrue(np.allclose(localvars.zt, zt_py, DIGITS))

	def pogs_check_convergence(self, lib, blas_handle, solver, f_list, g_list,
							   objectives, residuals, tolerances, settings,
							   localA, localvars):
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

		converged = lib.check_convergence(blas_handle, solver, objectives,
											  residuals, tolerances,
											  settings.gapstop, 0)

		self.load_all_local(lib, localvars, solver)
		obj_py = func_eval_python(g_list, localvars.x12)
		obj_py += func_eval_python(f_list, localvars.y12)
		obj_gap_py = abs(np.dot(localvars.z12, localvars.zt12))
		obj_dua_py = obj_py - obj_gap_py

		tol_primal = tolerances.atolm + (tolerances.reltol *
										 np.linalg.norm(localvars.y12))
		tol_dual = tolerances.atoln + (tolerances.reltol *
									   np.linalg.norm(localvars.xt12))

		self.assertAlmostEqual(significant_digits(objectives.gap),
							   significant_digits(obj_gap_py), DIGITS)
		self.assertAlmostEqual(significant_digits(tolerances.primal),
							   significant_digitis(tol_primal), DIGITS)
		self.assertAlmostEqual(significant_digits(tolerances.dual),
							   significant_digits(tol_dual), DIGITS)


		res_primal = np.linalg.norm(localA.dot(localvars.x12) -
									localvars.y12)
		res_dual = np.linalg.norm(localA.T.dot(localvars.yt12) +
								  localvars.xt12)

		self.assertAlmostEqual(significant_digits(residuals.primal),
							   significant_digits(res_primal), DIGITS)
		self.assertAlmostEqual(significant_digits(residuals.dual),
							   significant_digits(res_dual), DIGITS)
		self.assertAlmostEqual(significant_digits(residuals.gap),
							   significant_digits(abs(obj_gap_py)), DIGITS)

		converged_py = res_primal <= tolerances.primal and \
					   res_dual <= tolerances.dual

		self.assertEqual(converged, converged_py)

	def pogs_adapt_rho(self, lib, solver, settings, residuals,
					   tolerances, localvars):
		"""adaptive rho test

			rho is rescaled
			dual variable is rescaled accordingly

			the following equality should hold elementwise:

				rho_before * zt_before == rho_after * zt_after

			(here zt, or z tilde, is the dual variable)
		"""
		DIGITS = 7 - 2 * lib.FLOAT

		if settings.adaptiverho:
			rho_params = lib.adapt_params(1.05, 0, 0, 1)
			zt_before = localvars.zt
			rho_before = solver.contents.rho
			lib.adaptrho(solver, rho_params, residuals, tolerances, 1)
			self.load_all_local(lib, localvars, solver)
			zt_after = localvars.zt
			rho_after = solver.contents.rho
			self.assertTrue(np.allclose(rho_after * zt_after,
										rho_before * zt_before, DIGITS))

	def pogs_unscaling(self, lib, solver, output, localvars):
		"""pogs unscaling test

			solver variables are unscaled and copied to output.

			the following equalities should hold elementwise:

				x^{k+1/2} * e - x_out == 0
				y^{k+1/2} / d - y_out == 0
				-rho * xt^{k+1/2} / e - mu_out == 0
				-rho * yt^{k+1/2} * d - nu_out == 0
		"""
		DIGITS = 2 if lib.FLOAT else 3

		if not isinstance(solver, lib.pogs_solver_p):
			raise TypeError('argument "solver" must be of type {}'.format(
							lib.pogs_solver_p))

		if not isinstance(output, self.PogsOutputLocal):
			raise TypeError('argument "output" must be of type {}'.format(
							self.PogsOutputLocal))

		if not isinstance(localvars, self.PogsVariablesLocal):
			raise TypeError('argument "localvars" must be of type {}'.format(
							self.PogsVariablesLocal))

		self.load_all_local(lib, localvars, solver)
		lib.copy_output(solver, output.ptr)
		rho = solver.contents.rho

		self.assertTrue(np.allclose(localvars.x12 * localvars.e, output.x,
									DIGITS))
		self.assertTrue(np.allclose(localvars.y12, localvars.d * output.y,
									DIGITS))
		self.assertTrue(np.allclose(-rho * localvars.xt12,
									localvars.e * output.mu, DIGITS))
		self.assertTrue(np.allclose(-rho * localvars.yt12 * localvars.d,
									output.nu, DIGITS))



	def test_default_settings(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None,
										 None)

			lib.set_default_settings(settings)

			self.assertAlmostEqual(settings.alpha, ALPHA_DEFAULT, 3)
			self.assertAlmostEqual(settings.rho, RHO_DEFAULT, 3)
			self.assertAlmostEqual(settings.abstol, ABSTOL_DEFAULT, 3)
			self.assertAlmostEqual(settings.reltol, RELTOL_DEFAULT, 3)
			self.assertAlmostEqual(settings.maxiter, MAXITER_DEFAULT, 3)
			self.assertAlmostEqual(settings.verbose, VERBOSE_DEFAULT, 3)
			self.assertAlmostEqual(settings.suppress, SUPPRESS_DEFAULT, 3)
			self.assertAlmostEqual(settings.adaptiverho, ADAPTIVE_DEFAULT, 3)
			self.assertAlmostEqual(settings.gapstop, GAPSTOP_DEFAULT, 3)
			self.assertAlmostEqual(settings.warmstart, WARMSTART_DEFAULT, 3)
			self.assertAlmostEqual(settings.resume, RESUME_DEFAULT, 3)

	def test_pogs_init_finish(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				A = self.A_test.astype(lib.pyfloat)
				A_ptr = A.ctypes.data_as(lib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				lib.pogs_finish(solver, 1)

	def test_pogs_private_api(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			f = lib.function_vector(0, None)
			g = lib.function_vector(0, None)
			lib.function_vector_calloc(f, m)
			lib.function_vector_calloc(g, n)
			self.register_var('f', f, lib.function_vector_free)
			self.register_var('g', g, lib.function_vector_free)
			f_py = np.zeros(m).astype(lib.function)
			g_py = np.zeros(n).astype(lib.function)
			for i in xrange(m):
				f_py[i] = lib.function(lib.function_enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = lib.function(lib.function_enums.IndGe0, 1, 0, 1, 0, 0)
			f_list = [lib.function(*f_) for f_ in f_py]
			g_list = [lib.function(*g_) for g_ in g_py]


			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A += self.A_test
				A_ptr = A.ctypes.data_as(lib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = self.PogsOutputLocal(lib, m, n)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)


				localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)

				if lib.full_api_accessible:
					localA = np.zeros(
							(m, n), order=pyorder).astype(lib.pyfloat)
					localA_ptr = localA.ctypes.data_as(lib.ok_float_p)
					lib.matrix_memcpy_am(
							localA_ptr, solver.contents.M.contents.A, order)

					res = lib.pogs_residuals(0, 0, 0)
					tols = lib.make_tolerances(settings, m, n)
					obj = lib.pogs_objectives(0, 0, 0)


					self.pogs_equilibration(lib, solver, A, localA, localvars)
					self.pogs_projector(lib, hdl, solver, localA)
					self.pogs_scaling(lib, solver, f, f_py, g,
									  g_py, localvars)
					self.pogs_primal_update(lib, solver, localvars)
					self.pogs_prox(lib, hdl, solver, f, f_py, g,
								   g_py, localvars)
					self.pogs_primal_project(lib, hdl, solver, settings,
											 localA, localvars)
					self.pogs_dual_update(lib, hdl, solver, settings,
										  localvars)
					self.pogs_check_convergence(lib, hdl, solver, f_list,
												g_list, obj, res, tols,
												settings, localA, localvars)
					self.pogs_adapt_rho(lib, solver, settings, res, tols,
										localvars)
					self.pogs_unscaling(lib, solver, output, localvars)

			lib.pogs_finish(solver, 0)

			self.free_var('f')
			self.free_var('g')
			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_call(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			f = lib.function_vector(0, None)
			g = lib.function_vector(0, None)

			lib.function_vector_calloc(f, m)
			lib.function_vector_calloc(g, n)
			self.register_var('f', f, lib.function_vector_free)
			self.register_var('g', g, lib.function_vector_free)
			f_py = np.zeros(m).astype(lib.function)
			g_py = np.zeros(n).astype(lib.function)
			f_ptr = f_py.ctypes.data_as(lib.function_p)
			g_ptr = g_py.ctypes.data_as(lib.function_p)
			for i in xrange(m):
				f_py[i] = lib.function(lib.function_enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = lib.function(lib.function_enums.IndGe0, 1, 0, 1, 0, 0)

			lib.function_vector_memcpy_va(f, f_ptr)
			lib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A += self.A_test
				A_ptr = A.ctypes.data_as(lib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = self.PogsOutputLocal(lib, m, n)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)
				settings.verbose = 0

				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				lib.pogs_finish(solver, 0)

				if info.converged:
					rtol = settings.reltol
					atolm = settings.abstol * (m**0.5)
					atoln = settings.abstol * (n**0.5)
					y_norm = np.linalg.norm(output.y)
					mu_norm = np.linalg.norm(output.mu)

					primal_feas = np.linalg.norm(A.dot(output.x) - output.y)
					dual_feas = np.linalg.norm(A.T.dot(output.nu) + output.mu)

					self.assertTrue(primal_feas <= 10 * (atolm + rtol * y_norm))
					self.assertTrue(dual_feas <= 20 * (atoln + rtol * mu_norm))

			self.free_var('f')
			self.free_var('g')
			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_call_unified(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			f = lib.function_vector(0, None)
			g = lib.function_vector(0, None)

			lib.function_vector_calloc(f, m)
			lib.function_vector_calloc(g, n)
			self.register_var('f', f, lib.function_vector_free)
			self.register_var('g', g, lib.function_vector_free)
			f_py = np.zeros(m).astype(lib.function)
			g_py = np.zeros(n).astype(lib.function)
			f_ptr = f_py.ctypes.data_as(lib.function_p)
			g_ptr = g_py.ctypes.data_as(lib.function_p)
			for i in xrange(m):
				f_py[i] = lib.function(lib.function_enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = lib.function(lib.function_enums.IndGe0, 1, 0, 1, 0, 0)

			lib.function_vector_memcpy_va(f, f_ptr)
			lib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A += self.A_test
				A_ptr = A.ctypes.data_as(lib.ok_float_p)
				m, n = A.shape
				output = self.PogsOutputLocal(lib, m, n)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				settings.verbose = 0

				lib.set_default_settings(settings)
				lib.pogs(A_ptr, f, g, settings, info, output.ptr, order,
					lib.enums.EquilSinkhorn, 0)

				if info.converged:
					rtol = settings.reltol
					atolm = settings.abstol * (m**0.5)
					atoln = settings.abstol * (n**0.5)
					y_norm = np.linalg.norm(output.y)
					mu_norm = np.linalg.norm(output.mu)

					primal_feas = np.linalg.norm(A.dot(output.x) - output.y)
					dual_feas = np.linalg.norm(A.T.dot(output.nu) + output.mu)
					self.assertTrue(primal_feas <= 10 * (atolm + rtol * y_norm))
					self.assertTrue(dual_feas <= 20 * (atoln + rtol * mu_norm))

			self.free_var('f')
			self.free_var('g')

			self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_warmstart(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			x_rand = np.random.rand(n).astype(lib.pyfloat)
			nu_rand = np.random.rand(m).astype(lib.pyfloat)

			f = lib.function_vector(0, None)
			g = lib.function_vector(0, None)

			lib.function_vector_calloc(f, m)
			lib.function_vector_calloc(g, n)
			self.register_var('f', f, lib.function_vector_free)
			self.register_var('g', g, lib.function_vector_free)
			f_py = np.zeros(m).astype(lib.function)
			g_py = np.zeros(n).astype(lib.function)
			f_ptr = f_py.ctypes.data_as(lib.function_p)
			g_ptr = g_py.ctypes.data_as(lib.function_p)
			for i in xrange(m):
				f_py[i] = lib.function(lib.function_enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = lib.function(lib.function_enums.IndGe0, 1, 0, 1, 0, 0)

			lib.function_vector_memcpy_va(f, f_ptr)
			lib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A += self.A_test
				A_ptr = A.ctypes.data_as(lib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = self.PogsOutputLocal(lib, m, n)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)
				settings.maxiter = 0
				settings.x0 = x_rand.ctypes.data_as(lib.ok_float_p)
				settings.nu0 = nu_rand.ctypes.data_as(lib.ok_float_p)
				settings.warmstart = 1

				print "\nwarm start variable loading test (0 iters)"
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				if lib.full_api_accessible:
					# test warmstart feed

					rho = solver.contents.rho
					localA = np.zeros(
							(m, n), order=pyorder).astype(lib.pyfloat)
					localA_ptr = localA.ctypes.data_as(lib.ok_float_p)
					lib.matrix_memcpy_am(
							localA_ptr, solver.contents.M.contents.A, order)
					localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)
					self.load_all_local(lib, localvars, solver)
					self.assertTrue(np.allclose(x_rand,
												localvars.e * localvars.x,
												DIGITS))
					self.assertTrue(np.allclose(nu_rand * -1/rho,
												localvars.d * localvars.yt,
												DIGITS))
					self.assertTrue(np.allclose(localA.dot(localvars.x),
												localvars.y, DIGITS))
					self.assertTrue(np.allclose(localA.T.dot(localvars.yt),
												-localvars.xt, DIGITS))

				# COLD START
				print "\ncold"
				settings.warmstart = 0
				settings.maxiter = MAXITER_DEFAULT
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# REPEAT
				print "\nrepeat"
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# RESUME
				print "\nresume"
				settings.resume = 1
				settings.rho = info.rho
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# WARM START: x0
				print "\nwarm start x0"
				settings.resume = 0
				settings.rho = 1
				settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# WARM START: x0, rho
				print "\nwarm start x0, rho"
				settings.resume = 0
				settings.rho = info.rho
				settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# WARM START: x0, nu0
				print "\nwarm start x0, nu0"
				settings.resume = 0
				settings.rho = 1
				settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
				settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# WARM START: x0, nu0
				print "\nwarm start x0, nu0, rho"
				settings.resume = 0
				settings.rho = info.rho
				settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
				settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				lib.pogs_finish(solver, 0)

			self.free_var('f')
			self.free_var('g')
			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_io(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			x_rand = np.random.rand(n).astype(lib.pyfloat)
			nu_rand = np.random.rand(m).astype(lib.pyfloat)

			f = lib.function_vector(0, None)
			g = lib.function_vector(0, None)

			lib.function_vector_calloc(f, m)
			lib.function_vector_calloc(g, n)
			self.register_var('f', f, lib.function_vector_free)
			self.register_var('g', g, lib.function_vector_free)
			f_py = np.zeros(m).astype(lib.function)
			g_py = np.zeros(n).astype(lib.function)
			f_ptr = f_py.ctypes.data_as(lib.function_p)
			g_ptr = g_py.ctypes.data_as(lib.function_p)
			for i in xrange(m):
				f_py[i] = lib.function(lib.function_enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = lib.function(lib.function_enums.IndGe0, 1, 0, 1, 0, 0)

			lib.function_vector_memcpy_va(f, f_ptr)
			lib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), order=pyorder).astype(lib.pyfloat)
				A += self.A_test
				A_ptr = A.ctypes.data_as(lib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = self.PogsOutputLocal(lib, m, n)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)
				settings.verbose = 1

				print 'initial solve -> export data'
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				k_orig = info.k

				mindim = min(m, n)
				A_equil = np.zeros(
						(m, n), order=pyorder).astype(lib.pyfloat)
				A_equil_ptr = A_equil.ctypes.data_as(lib.ok_float_p)
				if lib.direct:
					LLT = np.zeros((mindim, mindim), order=pyorder)
					LLT = LLT.astype(lib.pyfloat)
					LLT_ptr = LLT.ctypes.data_as(lib.ok_float_p)
				else:
					LLT = c_void_p()
					LLT_ptr = LLT

				d = np.zeros(m).astype(lib.pyfloat)
				e = np.zeros(n).astype(lib.pyfloat)
				z = np.zeros(m + n).astype(lib.pyfloat)
				z12 = np.zeros(m + n).astype(lib.pyfloat)
				zt = np.zeros(m + n).astype(lib.pyfloat)
				zt12 = np.zeros(m + n).astype(lib.pyfloat)
				zprev = np.zeros(m + n).astype(lib.pyfloat)
				rho = np.zeros(1).astype(lib.pyfloat)

				d_ptr =d.ctypes.data_as(lib.ok_float_p)
				e_ptr = e.ctypes.data_as(lib.ok_float_p)
				z_ptr = z.ctypes.data_as(lib.ok_float_p)
				z12_ptr = z12.ctypes.data_as(lib.ok_float_p)
				zt_ptr = zt.ctypes.data_as(lib.ok_float_p)
				zt12_ptr = zt12.ctypes.data_as(lib.ok_float_p)
				zprev_ptr = zprev.ctypes.data_as(lib.ok_float_p)
				rho_ptr = rho.ctypes.data_as(lib.ok_float_p)

				lib.pogs_extract_solver(solver, A_equil_ptr, LLT_ptr, d_ptr,
										e_ptr, z_ptr, z12_ptr, zt_ptr,
										zt12_ptr, zprev_ptr, rho_ptr, order)


				lib.pogs_finish(solver, 0)


				solver = lib.pogs_load_solver(A_equil_ptr, LLT_ptr, d_ptr,
											  e_ptr, z_ptr, z12_ptr, zt_ptr,
											  zt12_ptr, zprev_ptr, rho[0], m,
											  n, order)

				settings.resume = 1
				print 'import data -> solve again'
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertTrue(info.k <= k_orig or not info.converged)
				lib.pogs_finish(solver, 0)

			self.free_var('f')
			self.free_var('g')
			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)