from numpy import zeros, array, ndarray
from numpy.linalg import norm
from numpy.random import rand
from optkit.utils.proxutils import prox_eval_python
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

	def load_all_local(self, lib, py_vars, solver):
		if not isinstance(py_vars, self.PogsVariablesLocal):
			raise TypeError('argument "py_vars" must be of type {}'.format(
							self.PogsVariablesLocal))
		elif 'pogs_solver_p' not in lib.__dict__:
			raise ValueError('argument "lib" must contain field named '
							 '"pogs_solver_p"')
		elif not isinstance(solver, lib.pogs_solver_p):
			raise TypeError('argument "solver" must be of type {}'.format(
							lib.pogs_solver_p))

		z = solver.contents.z
		self.load_to_local(lib, py_vars.z, z.contents.primal.contents.vec)
		self.load_to_local(lib, py_vars.z12, z.contents.primal12.contents.vec)
		self.load_to_local(lib, py_vars.zt, z.contents.dual.contents.vec)
		self.load_to_local(lib, py_vars.zt12, z.contents.dual12.contents.vec)
		self.load_to_local(lib, py_vars.prev, z.contents.prev.contents.vec)

		if 'pogs_matrix_p' in lib.__dict__:
			solver_work = solver.contents.M
		elif 'pogs_work_p' in lib.__dict__:
			solver_work = solver.contents.W
		else:
			raise ValueError('argument "lib" must contain a field named'
							 '"pogs_matrix_p" or "pogs_work_p"')

		self.load_to_local(lib, py_vars.d, solver_work.contents.d)
		self.load_to_local(lib, py_vars.e, solver_work.contents.e)

	def register_solver(self, name, solver, libcall):
		class solver_free(object):
			def __init__(self, libcall_):
				self.libcall = libcall_
			def __call__(self, solver):
				return self.libcall(solver, 0)
		self.register_var(name, solver, solver_free(libcall) )

	def gen_registered_pogs_fns(self, lib, m, n, name1='f', name2='g'):
		f, f_py, f_ptr = self.register_fnvector(lib, m, name1)
		g, g_py, g_ptr = self.register_fnvector(lib, n, name2)

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

	def assert_pogs_scaling(self, lib, solver, f, f_py, g, g_py, local_vars):
		m = len(f_py)
		n = len(g_py)
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

		def fv_list2arrays(function_vector_list):
			fv = function_vector_list
			fvh = array([fv_.h for fv_ in fv])
			fva = array([fv_.a for fv_ in fv])
			fvb = array([fv_.b for fv_ in fv])
			fvc = array([fv_.c for fv_ in fv])
			fvd = array([fv_.d for fv_ in fv])
			fve = array([fv_.e for fv_ in fv])
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
		self.load_all_local(lib, local_vars, solver)

		# scaled vars
		self.assertVecEqual( f_a0, local_vars.d * f_a1, ATOLM, RTOL )
		self.assertVecEqual( f_d0, local_vars.d * f_d1, ATOLM, RTOL )
		self.assertVecEqual( f_e0, local_vars.d * f_e1, ATOLM, RTOL )
		self.assertVecEqual( g_a0 * local_vars.e, g_a1, ATOLN, RTOL )
		self.assertVecEqual( g_d0 * local_vars.e, g_d1, ATOLN, RTOL )
		self.assertVecEqual( g_e0 * local_vars.e, g_e1, ATOLN, RTOL )

		# unchanged vars
		self.assertVecEqual( f_h0, f_h1, ATOLM, RTOL )
		self.assertVecEqual( f_b0, f_b1, ATOLM, RTOL )
		self.assertVecEqual( f_c0, f_c1, ATOLM, RTOL )
		self.assertVecEqual( g_h0, g_h1, ATOLN, RTOL )
		self.assertVecEqual( g_b0, g_b1, ATOLN, RTOL )
		self.assertVecEqual( g_c0, g_c1, ATOLN, RTOL )

	def assert_pogs_primal_update(self, lib, solver, local_vars):
		"""primal update test

			set

				z^k = z^{k-1}

			check

				z^k == z^{k-1}

			holds elementwise
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLMN = RTOL * (local_vars.m + local_vars.n)**0.5
		self.assertCall( lib.set_prev(solver.contents.z) )
		self.load_all_local(lib, local_vars, solver)
		self.assertVecEqual( local_vars.z, local_vars.prev, ATOLMN, RTOL )

	def assert_pogs_prox(self, lib, blas_handle, solver, f, f_py, g, g_py,
						 local_vars):
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

		z = solver.contents.z
		rho = solver.contents.rho
		self.assertCall( lib.prox(blas_handle, f, g, z, rho) )
		self.load_all_local(lib, local_vars, solver)

		f_list = [lib.function(*f_) for f_ in f_py]
		g_list = [lib.function(*f_) for f_ in g_py]
		for i in xrange(len(f_py)):
			f_list[i].a *= local_vars.d[i]
			f_list[i].d *= local_vars.d[i]
			f_list[i].e *= local_vars.d[i]
		for j in xrange(len(g_py)):
			g_list[j].a /= local_vars.e[j]
			g_list[j].d /= local_vars.e[j]
			g_list[j].e /= local_vars.e[j]

		x_arg = local_vars.x - local_vars.xt
		y_arg = local_vars.y - local_vars.yt
		x_out = prox_eval_python(g_list, rho, x_arg)
		y_out = prox_eval_python(f_list, rho, y_arg)
		self.assertVecEqual( local_vars.x12, x_out, ATOLN, RTOL )
		self.assertVecEqual( local_vars.y12, y_out, ATOLM, RTOL )

	def assert_pogs_dual_update(self, lib, blas_handle, solver, local_vars):
		"""dual update test

			set

				zt^{k+1/2} = z^{k+1/2} - z^{k} + zt^{k}
				zt^{k+1} = zt^{k} - z^{k+1} + alpha * z^{k+1/2} +
											  (1-alpha) * z^k

			in C and Python, check that results agree
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLMN = RTOL * (local_vars.m + local_vars.n)**0.5

		z = solver.contents.z

		self.load_all_local(lib, local_vars, solver)
		alpha = solver.contents.settings.contents.alpha
		zt12_py = local_vars.z12 - local_vars.prev + local_vars.zt
		zt_py = local_vars.zt - local_vars.z + (
					alpha * local_vars.z12 + (1-alpha) * local_vars.prev)

		self.assertCall(lib.update_dual(blas_handle, z, alpha) )
		self.load_all_local(lib, local_vars, solver)
		self.assertVecEqual( local_vars.zt12, zt12_py, ATOLMN, RTOL )
		self.assertVecEqual( local_vars.zt, zt_py, ATOLMN, RTOL )


	def assert_pogs_adapt_rho(self, lib, solver, residuals, tolerances,
							  local_vars):
		"""adaptive rho test

			rho is rescaled
			dual variable is rescaled accordingly

			the following equality should hold elementwise:

				rho_before * zt_before == rho_after * zt_after

			(here zt, or z tilde, is the dual variable)
		"""
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLMN = RTOL * (local_vars.m + local_vars.n)**0.5

		rho_p = lib.ok_float_p()
		rho_p.contents = lib.ok_float(solver.contents.rho)

		z = solver.contents.z
		settings = solver.contents.settings

		if settings.contents.adaptiverho:
			rho_params = lib.adapt_params(1.05, 0, 0, 1)
			zt_before = local_vars.zt
			rho_before = solver.contents.rho
			self.assertCall( lib.adaptrho(z, solver.contents.settings, rho_p,
										  rho_params, residuals, tolerances,
										  1) )
			self.load_all_local(lib, local_vars, solver)
			zt_after = local_vars.zt
			rho_after = rho_p.contents
			self.assertVecEqual( rho_after * zt_after,
									   rho_before * zt_before, ATOLMN, RTOL )

	def assert_pogs_convergence(self, A, settings, output, gpu=False,
								single_precision=False):
		if not isinstance(A, ndarray):
			raise TypeError('argument "A" must be of type {}'.format(ndarray))

		m, n = A.shape
		rtol = settings.reltol
		atolm = settings.abstol * (m**0.5)
		atoln = settings.abstol * (n**0.5)

		P = 10 * 1.5**int(single_precision) * 1.5**int(gpu);
		D = 20 * 1.5**int(single_precision) * 1.5**int(gpu);

		self.assertVecEqual(A.dot(output.x), output.y, atolm, P * rtol)
		self.assertVecEqual(A.T.dot(output.nu), -output.mu, atoln, D * rtol)

	def assert_pogs_unscaling(self, lib, output, solver, local_vars):
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
		ATOLM = RTOL * local_vars.m**0.5
		ATOLN = RTOL * local_vars.n**0.5

		if not isinstance(solver, lib.pogs_solver_p):
			raise TypeError('argument "solver" must be of type {}'.format(
							lib.pogs_solver_p))

		if not isinstance(output, self.PogsOutputLocal):
			raise TypeError('argument "output" must be of type {}'.format(
							self.PogsOutputLocal))

		if not isinstance(local_vars, self.PogsVariablesLocal):
			raise TypeError('argument "local_vars" must be of type {}'.format(
							self.PogsVariablesLocal))

		rho = solver.contents.rho
		suppress = solver.contents.settings.contents.suppress
		self.load_all_local(lib, local_vars, solver)

		if 'pogs_matrix_p' in lib.__dict__:
			solver_work = solver.contents.M
		elif 'pogs_work_p' in lib.__dict__:
			solver_work = solver.contents.W
		else:
			raise ValueError('argument "lib" must contain a field named'
							 '"pogs_matrix_p" or "pogs_work_p"')

		self.assertCall( lib.copy_output(
				output.ptr, solver.contents.z, solver_work.contents.d,
				solver_work.contents.e, rho, suppress) )

		self.assertVecEqual(
				local_vars.x12 * local_vars.e, output.x, ATOLN, RTOL )
		self.assertVecEqual(
				local_vars.y12, local_vars.d * output.y, ATOLM, RTOL )
		self.assertVecEqual(
				-rho * local_vars.xt12, local_vars.e * output.mu, ATOLN, RTOL )
		self.assertVecEqual(
				-rho * local_vars.yt12 * local_vars.d, output.nu, ATOLM, RTOL )

	def assert_warmstart_sequence(self, lib, solver, f, g, settings, info,
								  output):

		if not isinstance(solver, lib.pogs_solver_p):
			raise TypeError('argument "solver" must be of type {}'.format(
							lib.pogs_solver_p))

		settings.verbose = 1

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