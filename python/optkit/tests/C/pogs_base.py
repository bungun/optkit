from numpy import zeros, ndarray
from numpy.linalg import norm
from optkit.tests.C.base import OptkitCTestCase

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

class OptkitCPogsTestCase(OptkitCTestCase):
	class PogsVariablesLocal():
		def __init__(self, m, n, pytype):
			self.m = m
			self.n = n
			self.z = zeros(m + n).astype(pytype)
			self.z12 = zeros(m + n).astype(pytype)
			self.zt = zeros(m + n).astype(pytype)
			self.zt12 = zeros(m + n).astype(pytype)
			self.prev = zeros(m + n).astype(pytype)
			self.d = zeros(m).astype(pytype)
			self.e = zeros(n).astype(pytype)

		@property
		def x(self):
			return self.z[self.m:]

		@property
		def y(self):
			return self.z[:self.m]

		@property
		def x12(self):
			return self.z12[self.m:]

		@property
		def y12(self):
			return self.z12[:self.m]

		@property
		def xt(self):
			return self.zt[self.m:]

		@property
		def yt(self):
			return self.zt[:self.m]

		@property
		def xt12(self):
			return self.zt12[self.m:]

		@property
		def yt12(self):
			return self.zt12[:self.m]

	class PogsOutputLocal():
		def __init__(self, lib, m, n):
			self.x = zeros(n).astype(lib.pyfloat)
			self.y = zeros(m).astype(lib.pyfloat)
			self.mu = zeros(n).astype(lib.pyfloat)
			self.nu = zeros(m).astype(lib.pyfloat)
			self.ptr = lib.pogs_output(self.x.ctypes.data_as(lib.ok_float_p),
									   self.y.ctypes.data_as(lib.ok_float_p),
									   self.mu.ctypes.data_as(lib.ok_float_p),
									   self.nu.ctypes.data_as(lib.ok_float_p))

	def load_to_local(self, lib, py_vector, c_vector):
		self.assertCall( lib.vector_memcpy_av(
				py_vector.ctypes.data_as(lib.ok_float_p), c_vector, 1) )

	def load_all_local(self, lib, py_vars, solver_vars, solver_work):
		if not isinstance(py_vars, self.PogsVariablesLocal):
			raise TypeError('argument "py_vars" must be of type {}'.format(
							self.PogsVariablesLocal))

		z = solver_vars;
		self.load_to_local(lib, py_vars.z, z.contents.primal.contents.vec)
		self.load_to_local(lib, py_vars.z12, z.contents.primal12.contents.vec)
		self.load_to_local(lib, py_vars.zt, z.contents.dual.contents.vec)
		self.load_to_local(lib, py_vars.zt12, z.contents.dual12.contents.vec)
		self.load_to_local(lib, py_vars.prev, z.contents.prev.contents.vec)

		if solver_work is not None:
			self.load_to_local(lib, py_vars.d, solver_work.contents.d)
			self.load_to_local(lib, py_vars.e, solver_work.contents.e)

	def register_solver(self, name, solver, libcall):
		class solver_free(object):
			def __call__(self, solver):
				return libcall(solver, 0)
		self.register_var(name, solver, solver_free(lib.pogs_finish) )

	def gen_registered_pogs_test_vars(self, lib, m, n):
			f, f_py, f_ptr = self.register_fnvector(lib, m, 'f')
			g, g_py, g_ptr = self.register_fnvector(lib, n, 'g')

			for i in xrange(m):
				f_py[i] = lib.function(lib.function_enums.Abs, 1, 1, 1, 0, 0)
			for j in xrange(n):
				g_py[j] = lib.function(lib.function_enums.IndGe0, 1, 0, 1, 0, 0)

			self.assertCall( lib.function_vector_memcpy_va(f, f_ptr) )
			self.assertCall( lib.function_vector_memcpy_va(g, g_ptr) )

		return f, f_py, g, g_py

	def gen_pogs_params(self, lib, m, n):
		output = self.PogsOutputLocal(lib, m, n)
		info = lib.pogs_info()
		settings = lib.pogs_settings()
		self.assertCall( lib.set_default_settings(settings) )
		settings.verbose = self.VERBOSE_TEST
		return output, info, settings

	def assert_default_settings(self, lib):
		settings = lib.pogs_settings()

		TOL = 1e-3
		self.assertCall( lib.set_default_settings(settings) )
		self.assertScalarEqual(settings.alpha, ALPHA_DEFAULT, TOL )
		self.assertScalarEqual(settings.rho, RHO_DEFAULT, TOL )
		self.assertScalarEqual(settings.abstol, ABSTOL_DEFAULT, TOL )
		self.assertScalarEqual(settings.reltol, RELTOL_DEFAULT, TOL )
		self.assertScalarEqual(settings.maxiter, MAXITER_DEFAULT,
									 TOL )
		self.assertScalarEqual(settings.verbose, VERBOSE_DEFAULT,
									 TOL )
		self.assertScalarEqual(settings.suppress, SUPPRESS_DEFAULT,
									 TOL )
		self.assertScalarEqual(settings.adaptiverho,
									 ADAPTIVE_DEFAULT, TOL )
		self.assertScalarEqual(settings.gapstop, GAPSTOP_DEFAULT, TOL )
		self.assertScalarEqual(settings.warmstart, WARMSTART_DEFAULT,
									 TOL )
		self.assertScalarEqual(settings.resume, RESUME_DEFAULT, TOL )

	def assert_pogs_equilibration(self, lib, solver_work, A, A_equil):
		m, n = A.shape

		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS
		ATOLM = RTOL * m**0.5

		d_local = np.zeros(m).astype(lib.pyfloat)
		e_local = np.zeros(n).astype(lib.pyfloat)
		self.load_to_local(lib, solver_work.contents.d)
		self.load_to_local(lib, solver_work.contents.e)

		x_rand = np.random.rand(n)
		A_eqx = A_equil.dot(x_rand)
		DAEx = d_local * A.dot(e_local * x_rand)
		self.assertVecEqual( A_eqx, DAEx, ATOLM, RTOL )

	def assert_pogs_projector(self, lib, blas_handle, solver, A_equil):
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
					blas_handle, solver.contents.M.contents.P, x_in, y_in,
					x_out, y_out) )
		else:
			self.assertCall( lib.indirect_projector_project(
					blas_handle, solver.contents.M.contents.P, x_in, y_in,
					x_out, y_out) )

		self.load_to_local(lib, x_out_py, x_out)
		self.load_to_local(lib, y_out_py, y_out)

		self.assertVecEqual(
				A_equil.dot(x_out_py), y_out_py), ATOLM, RTOL )

		self.free_vars('x_in', 'y_in', 'x_out', 'y_out')

	def assert_pogs_scaling(self, lib, solver, f, f_py, g, g_py, localvars):
		m = len(f_py)
		n = len(g_py)
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

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
		self.assertCall( lib.function_vector_memcpy_va(f, f_py_ptr) )
		self.assertCall( lib.function_vector_memcpy_va(g, g_py_ptr) )

		# scale function vector
		self.assertCall( lib.update_problem(solver, f, g) )

		# retrieve scaled function vector parameters
		self.assertCall( lib.function_vector_memcpy_av(f_py_ptr,
													   solver.contents.f) )
		self.assertCall( lib.function_vector_memcpy_av(g_py_ptr,
													   solver.contents.g) )
		f_list = [lib.function(*f_) for f_ in f_py]
		g_list = [lib.function(*f_) for f_ in g_py]
		f_h1, f_a1, f_b1, f_c1, f_d1, f_e1 = fv_list2arrays(f_list)
		g_h1, g_a1, g_b1, g_c1, g_d1, g_e1 = fv_list2arrays(g_list)

		# retrieve scaling
		self.load_all_local(lib, localvars, solver.contents.z,
							solver.contents.M)

		# scaled vars
		self.assertVecEqual( f_a0, localvars.d * f_a1, ATOLM, RTOL )
		self.assertVecEqual( f_d0, localvars.d * f_d1, ATOLM, RTOL )
		self.assertVecEqual( f_e0, localvars.d * f_e1, ATOLM, RTOL )
		self.assertVecEqual( g_a0 * localvars.e, g_a1, ATOLN, RTOL )
		self.assertVecEqual( g_d0 * localvars.e, g_d1, ATOLN, RTOL )
		self.assertVecEqual( g_e0 * localvars.e, g_e1, ATOLN, RTOL )

		# unchanged vars
		self.assertVecEqual( f_h0, f_h1, ATOLM, RTOL )
		self.assertVecEqual( f_b0, f_b1, ATOLM, RTOL )
		self.assertVecEqual( f_c0, f_c1, ATOLM, RTOL )
		self.assertVecEqual( g_h0, g_h1, ATOLN, RTOL )
		self.assertVecEqual( g_b0, g_b1, ATOLN, RTOL )
		self.assertVecEqual( g_c0, g_c1, ATOLN, RTOL )

	def assert_pogs_warmstart(self, lib, rho, solver_vars, solver_work,
							  settings, localA, localvars):
		m, n = localA.shape
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

		x_rand = np.random.rand(n)
		nu_rand = np.random.rand(m)

		self.load_all_local(lib, localvars, solver_vars, solver_work)

		# check variable state is consistent after warm start
		self.assertVecEqual(
				x_rand, localvars.e * localvars.x, ATOLN, RTOL )
		self.assertVecEqual(
				nu_rand * -1/rho, localvars.d * localvars.yt,
				ATOLM, RTOL )
		self.assertVecEqual(
				localA.dot(localvars.x), localvars.y, ATOLM, RTOL )
		self.assertVecEqual(
				localA.T.dot(localvars.yt), -localvars.xt, ATOLN,
				RTOL )

	def assert_pogs_primal_update(self, lib, solver_vars, localvars):
		"""primal update test

			set

				z^k = z^{k-1}

			check

				z^k == z^{k-1}

			holds elementwise
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLMN = RTOL * (localvars.m + localvars.n)**0.5
		self.assertCall( lib.set_prev(solver_vars) )
		self.load_all_local(lib, localvars, solver_vars, None)
		self.assertVecEqual( localvars.z, localvars.prev, ATOLMN, RTOL )

	def assert_pogs_prox(self, lib, blas_handle, solver_vars, solver_work, rho,
						 f, f_py, g, g_py, localvars):
		"""proximal operator application test

			set

				x^{k+1/2} = prox_{g, rho}(x^k - xt^k)
				y^{k+1/2} = prox_{f, rho}(y^k - yt^k)

			in C and Python, check that results agree
		"""
		m = len(f_py)
		n = len(g_py)
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

		z = solver_vars
		self.assertCall( lib.prox(blas_handle, f, g, z, rho) )
		self.load_all_local(lib, localvars, z, solver_work)

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
		x_out = prox_eval_python(g_list, rho, x_arg)
		y_out = prox_eval_python(f_list, rho, y_arg)
		self.assertVecEqual( localvars.x12, x_out, ATOLN, RTOL )
		self.assertVecEqual( localvars.y12, y_out, ATOLM, RTOL )

	def assert_pogs_primal_project(self, lib, blas_handle, solver_vars,
								   solver_work, settings, localA, localvars):
		"""primal projection test

			set

				(x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})

			check that


				y^{k+1/2} == A * x^{k+1/2}

			holds to numerical tolerance
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * localvars.m**0.5

		self.assertCall( lib.project_primal(
				blas_handle, solver_work.contents.P, solver_vars,
				settings.alpha) )
		self.load_all_local(lib, localvars, solver_vars, None)
		self.assertVecEqual( localA.dot(localvars.x), localvars.y, ATOLM,
								   RTOL )

	def assert_pogs_check_convergence(self, lib, blas_handle, solver, f_list, g_list,
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
		RTOL = 10**(-DIGITS)

		converged = lib.check_convergence(blas_handle, solver, objectives,
										  residuals, tolerances)

		self.load_all_local(lib, localvars, solver.contents.z, None)
		obj_py = func_eval_python(g_list, localvars.x12)
		obj_py += func_eval_python(f_list, localvars.y12)
		obj_gap_py = abs(np.dot(localvars.z12, localvars.zt12))
		obj_dua_py = obj_py - obj_gap_py

		tol_primal = tolerances.atolm + (
				tolerances.reltol * np.linalg.norm(localvars.y12))
		tol_dual = tolerances.atoln + (
				tolerances.reltol * np.linalg.norm(localvars.xt12))

		self.assertScalarEqual( objectives.gap, obj_gap_py, RTOL )
		self.assertScalarEqual( tolerances.primal, tol_primal, RTOL )
		self.assertScalarEqual( tolerances.dual, tol_dual, RTOL )

		res_primal = np.linalg.norm(
				localA.dot(localvars.x12) - localvars.y12)
		res_dual = np.linalg.norm(
				localA.T.dot(localvars.yt12) + localvars.xt12)

		self.assertScalarEqual( residuals.primal, res_primal, RTOL )
		self.assertScalarEqual( residuals.dual, res_dual, RTOL )
		self.assertScalarEqual( residuals.gap, abs(obj_gap_py), RTOL )

		converged_py = res_primal <= tolerances.primal and \
					   res_dual <= tolerances.dual

		self.assertEqual( converged, converged_py )

	def assert_pogs_dual_update(self, lib, blas_handle, solvervars,
								settings, localvars):
		"""dual update test

			set

				zt^{k+1/2} = z^{k+1/2} - z^{k} + zt^{k}
				zt^{k+1} = zt^{k} - z^{k+1} + alpha * z^{k+1/2} +
											  (1-alpha) * z^k

			in C and Python, check that results agree
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLMN = RTOL * (localvars.m + localvars.n)**0.5

		z = solvervars

		self.load_all_local(lib, localvars, z, None)
		alpha = settings.alpha
		zt12_py = localvars.z12 - localvars.prev + localvars.zt
		zt_py = localvars.zt - localvars.z + (
					alpha * localvars.z12 + (1-alpha) * localvars.prev)

		self.assertCall(lib.update_dual(blas_handle, z, alpha) )
		self.load_all_local(lib, localvars, z, None)
		self.assertVecEqual( localvars.zt12, zt12_py, ATOLMN, RTOL )
		self.assertVecEqual( localvars.zt, zt_py, ATOLMN, RTOL )

	def assert_pogs_adapt_rho(self, lib, solvervars, settings, residuals,
					   tolerances, localvars):
		"""adaptive rho test

			rho is rescaled
			dual variable is rescaled accordingly

			the following equality should hold elementwise:

				rho_before * zt_before == rho_after * zt_after

			(here zt, or z tilde, is the dual variable)
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLMN = RTOL * (localvars.m + localvars.n)**0.5

		if settings.adaptiverho:
			rho_params = lib.adapt_params(1.05, 0, 0, 1)
			zt_before = localvars.zt
			rho_before = solver.contents.rho
			self.assertCall( lib.adaptrho(solver, rho_params, residuals,
										  tolerances, 1) )
			self.load_all_local(lib, localvars, solvervars, None)
			zt_after = localvars.zt
			rho_after = solver.contents.rho
			self.assertVecEqual( rho_after * zt_after,
									   rho_before * zt_before, ATOLMN, RTOL )

	def assert_pogs_convergence(self, A, settings, output):
		if not isinstance(A, ndarray):
			raise TypeError('argument "A" must be of type {}'.format(ndarray))

		m, n = A.shape
		rtol = settings.reltol
		atolm = settings.abstol * (m**0.5)
		atoln = settings.abstol * (n**0.5)
		y_norm = norm(output.y)
		mu_norm = norm(output.mu)

		primal_feas = norm(A.dot(output.x) - output.y)
		dual_feas = norm(A.T.dot(output.nu) + output.mu)

		self.assertTrue(primal_feas <= 10 * (atolm + rtol * y_norm))
		self.assertTrue(dual_feas <= 20 * (atoln + rtol * mu_norm))

	def assert_pogs_unscaling(self, lib, solver, output, localvars):
		"""pogs unscaling test

			solver variables are unscaled and copied to output.

			the following equalities should hold elementwise:

				x^{k+1/2} * e - x_out == 0
				y^{k+1/2} / d - y_out == 0
				-rho * xt^{k+1/2} / e - mu_out == 0
				-rho * yt^{k+1/2} * d - nu_out == 0
		"""
		DIGITS = 2 if lib.FLOAT else 3
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * localvars.m**0.5
		ATOLN = RTOL * localvars.n**0.5

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
		rho = solver.contents.rho

		self.assertCall(
				lib.copy_output(solver, output.ptr, rho,
								solver.contents.settings.contents.suppress) )

		self.assertVecEqual(
				localvars.x12 * localvars.e, output.x, ATOLN, RTOL )
		self.assertVecEqual(
				localvars.y12, localvars.d * output.y, ATOLM, RTOL )
		self.assertVecEqual(
				-rho * localvars.xt12, ocalvars.e * output.mu, ATOLN, RTOL )
		self.assertVecEqual(
				-rho * localvars.yt12 * localvars.d, output.nu, ATOLM, RTOL )

	def assert_warmstart_sequence(self, lib, solver, f, g, settings, info,
								  output):
		# COLD START
		print "\ncold"
		settings.warmstart = 0
		settings.maxiter = MAXITER_DEFAULT
		self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
						 output.ptr) )
		self.assertEqual( info.err, 0 )
		self.assertTrue( info.converged or info.k >= settings.maxiter )

		# REPEAT
		print "\nrepeat"
		self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
						 output.ptr) )
		self.assertEqual( info.err, 0 )
		self.assertTrue( info.converged or info.k >= settings.maxiter )

		# RESUME
		print "\nresume"
		settings.resume = 1
		settings.rho = info.rho
		self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
						 output.ptr) )
		self.assertEqual( info.err, 0 )
		self.assertTrue( info.converged or info.k >= settings.maxiter )

		# WARM START: x0
		print "\nwarm start x0"
		settings.resume = 0
		settings.rho = 1
		settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
		settings.warmstart = 1
		self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
						 output.ptr) )
		self.assertEqual( info.err, 0 )
		self.assertTrue( info.converged or info.k >= settings.maxiter )

		# WARM START: x0, rho
		print "\nwarm start x0, rho"
		settings.resume = 0
		settings.rho = info.rho
		settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
		settings.warmstart = 1
		self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
						 output.ptr) )
		self.assertEqual( info.err, 0 )
		self.assertTrue( info.converged or info.k >= settings.maxiter )

		# WARM START: x0, nu0
		print "\nwarm start x0, nu0"
		settings.resume = 0
		settings.rho = 1
		settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
		settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
		settings.warmstart = 1
		self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
						 output.ptr) )
		self.assertEqual( info.err, 0 )
		self.assertTrue( info.converged or info.k >= settings.maxiter )

		# WARM START: x0, nu0
		print "\nwarm start x0, nu0, rho"
		settings.resume = 0
		settings.rho = info.rho
		settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
		settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
		settings.warmstart = 1
		self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
						 output.ptr) )
		self.assertEqual( info.err, 0 )
		self.assertTrue( info.converged or info.k >= settings.maxiter )