import unittest
import os
import numpy as np
from ctypes import c_void_p, byref, cast, addressof
from optkit.libs import DenseLinsysLibs, ProxLibs, PogsLibs
from optkit.tests.defs import CONDITIONS
from optkit.utils.proxutils import func_eval_python, prox_eval_python

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

class PogsLibsTestCase(unittest.TestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.prox_libs = ProxLibs()
		self.pogs_libs = PogsLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		dlibs = []
		pxlibs = []
		libs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(
					self.dense_libs.get(single_precision=single_precision,
										gpu=gpu))
			pxlibs.append(
					self.prox_libs.get(dlibs[-1],
									   single_precision=single_precision,
									   gpu=gpu))
			libs.append(
					self.pogs_libs.get(dlibs[-1], pxlibs[-1],
									   single_precision=single_precision,
									   gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(pxlibs))
		self.assertTrue(any(libs))

class PogsVariablesLocal():
	def __init__(self, m, n, pytype):
		self.m = m
		self.n = n
		self.z = np.zeros(m + n, dtype=pytype)
		self.z12 = np.zeros(m + n, dtype=pytype)
		self.zt = np.zeros(m + n, dtype=pytype)
		self.zt12 = np.zeros(m + n, dtype=pytype)
		self.prev = np.zeros(m + n, dtype=pytype)
		self.d = np.zeros(m, dtype=pytype)
		self.e = np.zeros(n, dtype=pytype)

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
	def __init__(self, denselib, pogslib, m, n, pytype):
		self.x = np.zeros(n, dtype=pytype)
		self.y = np.zeros(m, dtype=pytype)
		self.mu = np.zeros(n, dtype=pytype)
		self.nu = np.zeros(m, dtype=pytype)
		self.ptr = pogslib.pogs_output(
				cast(self.x.ctypes.get_data(), denselib.ok_float_p),
				cast(self.y.ctypes.get_data(), denselib.ok_float_p),
				cast(self.mu.ctypes.get_data(), denselib.ok_float_p),
				cast(self.nu.ctypes.get_data(), denselib.ok_float_p))

class PogsTestCase(unittest.TestCase):
	"""TODO: docstring"""


	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.prox_libs = ProxLibs()
		self.pogs_libs = PogsLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = (4, 8)
		# self.shape = (150, 250)
		self.A_test = np.random.rand(*self.shape)
		self.x_test = np.random.rand(self.shape[1])


	def load_to_local(self, denselib, py_vector, c_vector):
		denselib.vector_memcpy_av(
				py_vector.ctypes.data_as(denselib.ok_float_p), c_vector, 1)

	def load_all_local(self, denselib, py_vars, solver):
		if not isinstance(py_vars, PogsVariablesLocal):
			raise TypeError('argument "py_vars" must be of type {}'.format(
							PogsVariablesLocal))

		self.load_to_local(denselib, py_vars.z,
						   solver.contents.z.contents.primal.contents.vec)
		self.load_to_local(denselib, py_vars.z12,
						   solver.contents.z.contents.primal12.contents.vec)
		self.load_to_local(denselib, py_vars.zt,
						   solver.contents.z.contents.dual.contents.vec)
		self.load_to_local(denselib, py_vars.zt12,
						   solver.contents.z.contents.dual12.contents.vec)
		self.load_to_local(denselib, py_vars.prev,
						   solver.contents.z.contents.prev.contents.vec)
		self.load_to_local(denselib, py_vars.d, solver.contents.M.contents.d)
		self.load_to_local(denselib, py_vars.e, solver.contents.M.contents.e)


	def pogs_equilibration(self, denselib, solver, A, localA, localvars,
						   pytype):
		m, n = A.shape
		d_local = np.zeros(m, dtype=pytype)
		e_local = np.zeros(n, dtype=pytype)
		self.load_to_local(denselib, d_local, solver.contents.M.contents.d)
		self.load_to_local(denselib, e_local, solver.contents.M.contents.e)

		x_rand = np.random.rand(n)
		A_eqx = localA.dot(x_rand)
		DAEx = d_local * A.dot(e_local * x_rand)
		self.assertTrue(np.allclose(A_eqx, DAEx))

	def pogs_projector(self, denselib, pogslib, blas_handle, solver, localA,
					   pytype):

		m, n = localA.shape

		x_in = denselib.vector(0, 0, None)
		y_in = denselib.vector(0, 0, None)
		x_out = denselib.vector(0, 0, None)
		y_out = denselib.vector(0, 0, None)

		denselib.vector_calloc(x_in, n)
		denselib.vector_calloc(y_in, m)
		denselib.vector_calloc(x_out, n)
		denselib.vector_calloc(y_out, m)


		x_in_py = np.random.rand(n).astype(pytype)
		y_in_py = np.random.rand(m).astype(pytype)
		x_out_py = np.zeros(n, dtype=pytype)
		y_out_py = np.zeros(m, dtype=pytype)

		x_in_ptr = x_in_py.ctypes.data_as(denselib.ok_float_p)
		y_in_ptr = y_in_py.ctypes.data_as(denselib.ok_float_p)

		denselib.vector_memcpy_va(x_in, x_in_ptr, 1)
		denselib.vector_memcpy_va(y_in, y_in_ptr, 1)


		if pogslib.direct:
			pogslib.direct_projector_project(blas_handle,
											 solver.contents.M.contents.P,
											 x_in, y_in, x_out, y_out)
		else:
			pogslib.indirect_projector_project(blas_handle,
											 solver.contents.M.contents.P,
											 x_in, y_in, x_out, y_out)

		self.load_to_local(denselib, x_out_py, x_out)
		self.load_to_local(denselib, y_out_py, y_out)

		self.assertTrue(np.allclose(localA.dot(x_out_py), y_out_py))

		denselib.vector_free(x_in)
		denselib.vector_free(y_in)
		denselib.vector_free(x_out)
		denselib.vector_free(y_out)

	def pogs_scaling(self, denselib, proxlib, pogslib, solver, f, f_py, g,
					 g_py, localvars):

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
		f_list = [proxlib.function(*f_) for f_ in f_py]
		g_list = [proxlib.function(*f_) for f_ in g_py]
		f_h0, f_a0, f_b0, f_c0, f_d0, f_e0 = fv_list2arrays(f_list)
		g_h0, g_a0, g_b0, g_c0, g_d0, g_e0 = fv_list2arrays(g_list)

		# copy function vector
		f_py_ptr = f_py.ctypes.data_as(proxlib.function_p)
		g_py_ptr = g_py.ctypes.data_as(proxlib.function_p)
		proxlib.function_vector_memcpy_va(f, f_py_ptr)
		proxlib.function_vector_memcpy_va(g, g_py_ptr)

		# scale function vector
		pogslib.update_problem(solver, f, g)

		# retrieve scaled function vector parameters
		proxlib.function_vector_memcpy_av(f_py_ptr, solver.contents.f)
		proxlib.function_vector_memcpy_av(g_py_ptr, solver.contents.g)
		f_list = [proxlib.function(*f_) for f_ in f_py]
		g_list = [proxlib.function(*f_) for f_ in g_py]
		f_h1, f_a1, f_b1, f_c1, f_d1, f_e1 = fv_list2arrays(f_list)
		g_h1, g_a1, g_b1, g_c1, g_d1, g_e1 = fv_list2arrays(g_list)


		# retrieve scaling
		self.load_all_local(denselib, localvars, solver)

		# scaled vars
		self.assertTrue(np.allclose(f_a0, localvars.d * f_a1))
		self.assertTrue(np.allclose(f_d0, localvars.d * f_d1))
		self.assertTrue(np.allclose(f_e0, localvars.d * f_e1))
		self.assertTrue(np.allclose(g_a0 * localvars.e, g_a1))
		self.assertTrue(np.allclose(g_d0 * localvars.e, g_d1))
		self.assertTrue(np.allclose(g_e0 * localvars.e, g_e1))

		# unchanged vars
		self.assertTrue(np.allclose(f_h0, f_h1))
		self.assertTrue(np.allclose(f_b0, f_b1))
		self.assertTrue(np.allclose(f_c0, f_c1))
		self.assertTrue(np.allclose(g_h0, g_h1))
		self.assertTrue(np.allclose(g_b0, g_b1))
		self.assertTrue(np.allclose(g_c0, g_c1))

	def pogs_warmstart(self, denselib, pogslib, solver, settings, localA,
					   localvars, pytype):

		m, n = localA.shape

		rho = solver.contents.rho

		x_rand = np.random.rand(n).astype(pytype)
		nu_rand = np.random.rand(m).astype(pytype)

		x_ptr = x_rand.ctypes.data_as(denselib.ok_float_p)
		nu_ptr = nu_rand.ctypes.data_as(denselib.ok_float_p)

		settings.x0 = x_ptr
		settings.nu0 = nu_ptr
		pogslib.update_settings(solver.contents.settings, settings)
		print "wheres the"

		pogslib.initialize_variables(solver)
		self.load_all_local(denselib, localvars, solver)

		print localvars.x
		print x_rand

		self.assertTrue(np.allclose(x_rand, localvars.e * localvars.x))
		self.assertTrue(np.allclose(nu_rand * localvars.d * -1/rho,
									localvars.yt))

		self.assertTrue(np.allclose(localA.dot(localvars.x), localvars.y))
		self.assertTrue(np.allclose(localA.T.dot(localvars.yt), -localvars.xt))


	def pogs_primal_update(self, denselib, pogslib, solver, localvars):
		"""primal update test

			set

				z^k = z^{k-1}

			check

				z^k == z^{k-1}

			holds elementwise
		"""
		pogslib.set_prev(solver.contents.z)
		self.load_all_local(denselib, localvars, solver)
		self.assertTrue(np.allclose(localvars.z, localvars.prev))


	def pogs_prox(self, denselib, proxlib, pogslib, blas_handle, solver, f,
				  f_py, g, g_py, localvars):
		"""proximal operator application test

			set

				x^{k+1/2} = prox_{g, rho}(x^k - xt^k)
				y^{k+1/2} = prox_{f, rho}(y^k - yt^k)

			in C and Python, check that results agree
		"""


		pogslib.prox(blas_handle, f, g, solver.contents.z,
						 solver.contents.rho)
		self.load_all_local(denselib, localvars, solver)

		f_list = [proxlib.function(*f_) for f_ in f_py]
		g_list = [proxlib.function(*f_) for f_ in g_py]
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
		self.assertTrue(np.allclose(localvars.x12, x_out))
		self.assertTrue(np.allclose(localvars.y12, y_out))

	def pogs_primal_project(self, denselib, pogslib, blas_handle, solver,
							settings, localA, localvars):
		"""primal projection test

			set

				(x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})

			check that


				y^{k+1/2} == A * x^{k+1/2}

			holds to numerical tolerance
		"""

		pogslib.project_primal(blas_handle, solver.contents.M.contents.P,
							   solver.contents.z, settings.alpha)
		self.load_all_local(denselib, localvars, solver)
		self.assertTrue(np.allclose(localA.dot(localvars.x), localvars.y))

	def pogs_dual_update(self, denselib, pogslib, blas_handle, solver,
						 settings, localvars):
		"""dual update test

			set

				zt^{k+1/2} = z^{k+1/2} - z^{k} + zt^{k}
				zt^{k+1} = zt^{k} - z^{k+1} + alpha * z^{k+1/2} +
											  (1-alpha) * z^k

			in C and Python, check that results agree
		"""
		self.load_all_local(denselib, localvars, solver)
		alpha = settings.alpha
		zt12_py = localvars.z12 - localvars.prev + localvars.zt
		zt_py = localvars.zt - localvars.z + (alpha * localvars.z12 +
								 			   (1-alpha) * localvars.prev)

		pogslib.update_dual(blas_handle, solver.contents.z, alpha)
		self.load_all_local(denselib, localvars, solver)
		self.assertTrue(np.allclose(localvars.zt12, zt12_py))
		self.assertTrue(np.allclose(localvars.zt, zt_py))

	def pogs_check_convergence(self, denselib, pogslib, blas_handle, solver,
							   f_list, g_list, objectives, residuals,
							   tolerances, settings, localA, localvars):
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


		converged = pogslib.check_convergence(blas_handle, solver, objectives,
											  residuals, tolerances,
											  settings.gapstop, 0)

		self.load_all_local(denselib, localvars, solver)
		obj_py = func_eval_python(g_list, localvars.x12)
		obj_py += func_eval_python(f_list, localvars.y12)
		obj_gap_py = abs(np.dot(localvars.z12, localvars.zt12))
		obj_dua_py = obj_py - obj_gap_py

		tol_primal = tolerances.atolm + (tolerances.reltol *
										 np.linalg.norm(localvars.y12))
		tol_dual = tolerances.atoln + (tolerances.reltol *
									   np.linalg.norm(localvars.xt12))

		self.assertAlmostEqual(objectives.gap, obj_gap_py)
		self.assertAlmostEqual(tolerances.primal, tol_primal)
		self.assertAlmostEqual(tolerances.dual, tol_dual)


		res_primal = np.linalg.norm(localA.dot(localvars.x12) -
									localvars.y12)
		res_dual = np.linalg.norm(localA.T.dot(localvars.yt12) +
								  localvars.xt12)

		self.assertAlmostEqual(residuals.primal, res_primal)
		self.assertAlmostEqual(residuals.dual, res_dual)
		self.assertAlmostEqual(residuals.gap, abs(obj_gap_py))

		converged_py = res_primal <= tolerances.primal and \
					   res_dual <= tolerances.dual

		self.assertEqual(converged, converged_py)

	def pogs_adapt_rho(self, denselib, pogslib, solver, settings, residuals,
					   tolerances, localvars):
		"""adaptive rho test

			rho is rescaled
			dual variable is rescaled accordingly

			the following equality should hold elementwise:

				rho_before * zt_before == rho_after * zt_after

			(here zt, or z tilde, is the dual variable)
		"""

		if settings.adaptiverho:
			rho_params = pogslib.adapt_params(1.05, 0, 0, 1)
			zt_before = localvars.zt
			rho_before = solver.contents.rho
			pogslib.adaptrho(solver, rho_params, residuals, tolerances, 1)
			self.load_all_local(denselib, localvars, solver)
			zt_after = localvars.zt
			rho_after = solver.contents.rho
			self.assertTrue(np.allclose(rho_after * zt_after,
										rho_before * zt_before))

	def pogs_unscaling(self, denselib, pogslib, solver, output, localvars):
		"""pogs unscaling test

			solver variables are unscaled and copied to output.

			the following equalities should hold elementwise:

				x^{k+1/2} * e - x_out == 0
				y^{k+1/2} / d - y_out == 0
				-rho * xt^{k+1/2} / e - mu_out == 0
				-rho * yt^{k+1/2} * d - nu_out == 0
		"""

		if not isinstance(solver, pogslib.pogs_solver_p):
			raise TypeError('argument "solver" must be of type {}'.format(
							pogslib.pogs_solver_p))

		if not isinstance(output, PogsOutputLocal):
			raise TypeError('argument "output" must be of type {}'.format(
							PogsOutputLocal))

		if not isinstance(localvars, PogsVariablesLocal):
			raise TypeError('argument "localvars" must be of type {}'.format(
							PogsVariablesLocal))

		self.load_all_local(denselib, localvars, solver)
		pogslib.copy_output(solver, output.ptr)
		rho = solver.contents.rho

		self.assertTrue(np.allclose(localvars.x12 * localvars.e, output.x, 3))
		self.assertTrue(np.allclose(localvars.y12, localvars.d * output.y, 3))
		self.assertTrue(np.allclose(-rho * localvars.xt12,
									localvars.e * output.mu, 3))
		self.assertTrue(np.allclose(-rho * localvars.yt12 * localvars.d,
									output.nu, 3))



	def test_default_settings(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			pxlib = self.prox_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			lib = self.pogs_libs.get(dlib, pxlib,
									 single_precision=single_precision,
									 gpu=gpu)

			if lib is None:
				continue

			settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None,
										 None)

			lib.set_default_settings(settings)

			self.assertEqual(settings.alpha, ALPHA_DEFAULT)
			self.assertEqual(settings.rho, RHO_DEFAULT)
			self.assertEqual(settings.abstol, ABSTOL_DEFAULT)
			self.assertEqual(settings.reltol, RELTOL_DEFAULT)
			self.assertEqual(settings.maxiter, MAXITER_DEFAULT)
			self.assertEqual(settings.verbose, VERBOSE_DEFAULT)
			self.assertEqual(settings.suppress, SUPPRESS_DEFAULT)
			self.assertEqual(settings.adaptiverho, ADAPTIVE_DEFAULT)
			self.assertEqual(settings.gapstop, GAPSTOP_DEFAULT)
			self.assertEqual(settings.warmstart, WARMSTART_DEFAULT)
			self.assertEqual(settings.resume, RESUME_DEFAULT)

	def test_pogs_init_finish(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			pxlib = self.prox_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			lib = self.pogs_libs.get(dlib, pxlib,
									 single_precision=single_precision,
									 gpu=gpu)

			if lib is None:
				continue

			pytype = np.float32 if single_precision else np.float64

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor

				A = self.A_test.astype(pytype)
				A_ptr = A.ctypes.data_as(dlib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				lib.pogs_finish(solver, 1)

	def test_pogs_private_api(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			pxlib = self.prox_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			lib = self.pogs_libs.get(dlib, pxlib,
									 single_precision=single_precision,
									 gpu=gpu)

			if lib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)
			pytype = np.float32 if single_precision else np.float64

			f = pxlib.function_vector(0, None)
			g = pxlib.function_vector(0, None)
			pxlib.function_vector_calloc(f, m)
			pxlib.function_vector_calloc(g, n)
			f_py = np.zeros(m, dtype=pxlib.function)
			g_py = np.zeros(n, dtype=pxlib.function)
			for i in xrange(m):
				f_py[i] = pxlib.function(pxlib.enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = pxlib.function(pxlib.enums.IndGe0, 1, 0, 1, 0, 0)
			f_list = [pxlib.function(*f_) for f_ in f_py]
			g_list = [pxlib.function(*g_) for g_ in g_py]


			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), dtype=pytype, order=pyorder)
				A += self.A_test
				A_ptr = A.ctypes.data_as(dlib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = PogsOutputLocal(dlib, lib, m, n, pytype)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)


				localvars = PogsVariablesLocal(m, n, pytype)

				if lib.full_api_accessible:
					localA = np.zeros((m, n), dtype=pytype, order=pyorder)
					localA_ptr = localA.ctypes.data_as(dlib.ok_float_p)
					dlib.matrix_memcpy_am(localA_ptr,
										  solver.contents.M.contents.A, order)

					res = lib.pogs_residuals(0, 0, 0)
					tols = lib.make_tolerances(settings, m, n)
					obj = lib.pogs_objectives(0, 0, 0)


					self.pogs_equilibration(dlib, solver, A, localA, localvars,
										    pytype)
					self.pogs_projector(dlib, lib, hdl, solver, localA, pytype)
					self.pogs_scaling(dlib, pxlib, lib, solver, f, f_py, g,
									  g_py, localvars)
					# self.pogs_warmstart(dlib, lib, solver, settings, localA,
										# localvars, pytype)
					self.pogs_primal_update(dlib, lib, solver, localvars)
					self.pogs_prox(dlib, pxlib, lib, hdl, solver, f, f_py, g,
								   g_py, localvars)
					self.pogs_primal_project(dlib, lib, hdl, solver, settings,
											 localA, localvars)
					self.pogs_dual_update(dlib, lib, hdl, solver, settings,
										  localvars)
					self.pogs_check_convergence(dlib, lib, hdl, solver, f_list,
												g_list, obj, res, tols,
												settings, localA, localvars)
					self.pogs_adapt_rho(dlib, lib, solver, settings, res, tols,
										localvars)
					self.pogs_unscaling(dlib, lib, solver, output, localvars)

			lib.pogs_finish(solver, 0)

			pxlib.function_vector_free(f)
			pxlib.function_vector_free(g)
			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_pogs_call(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			pxlib = self.prox_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			lib = self.pogs_libs.get(dlib, pxlib,
									 single_precision=single_precision,
									 gpu=gpu)

			if lib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)
			pytype = np.float32 if single_precision else np.float64

			f = pxlib.function_vector(0, None)
			g = pxlib.function_vector(0, None)

			pxlib.function_vector_calloc(f, m)
			pxlib.function_vector_calloc(g, n)
			f_py = np.zeros(m, dtype=pxlib.function)
			g_py = np.zeros(n, dtype=pxlib.function)
			f_ptr = f_py.ctypes.data_as(pxlib.function_p)
			g_ptr = g_py.ctypes.data_as(pxlib.function_p)
			for i in xrange(m):
				f_py[i] = pxlib.function(pxlib.enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = pxlib.function(pxlib.enums.IndGe0, 1, 0, 1, 0, 0)

			pxlib.function_vector_memcpy_va(f, f_ptr)
			pxlib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), dtype=pytype, order=pyorder)
				A += self.A_test
				A_ptr = A.ctypes.data_as(dlib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = PogsOutputLocal(dlib, lib, m, n, pytype)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)
				settings.verbose = 0

				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				lib.pogs_finish(solver, 0)

				if info.converged:
					primal_feas = A.dot(output.x) - output.y
					dual_feas = A.T.dot(output.nu) + output.mu
					self.assertTrue(all(primal_feas <= 10 * settings.reltol))
					self.assertTrue(all(dual_feas <= 10 * settings.reltol))

			pxlib.function_vector_free(f)
			pxlib.function_vector_free(g)
			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_pogs_call_unified(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			pxlib = self.prox_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			lib = self.pogs_libs.get(dlib, pxlib,
									 single_precision=single_precision,
									 gpu=gpu)

			if lib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)
			pytype = np.float32 if single_precision else np.float64

			f = pxlib.function_vector(0, None)
			g = pxlib.function_vector(0, None)

			pxlib.function_vector_calloc(f, m)
			pxlib.function_vector_calloc(g, n)
			f_py = np.zeros(m, dtype=pxlib.function)
			g_py = np.zeros(n, dtype=pxlib.function)
			f_ptr = f_py.ctypes.data_as(pxlib.function_p)
			g_ptr = g_py.ctypes.data_as(pxlib.function_p)
			for i in xrange(m):
				f_py[i] = pxlib.function(pxlib.enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = pxlib.function(pxlib.enums.IndGe0, 1, 0, 1, 0, 0)

			pxlib.function_vector_memcpy_va(f, f_ptr)
			pxlib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), dtype=pytype, order=pyorder)
				A += self.A_test
				A_ptr = A.ctypes.data_as(dlib.ok_float_p)
				m, n = A.shape
				output = PogsOutputLocal(dlib, lib, m, n, pytype)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				settings.verbose = 0

				lib.set_default_settings(settings)
				lib.pogs(A_ptr, f, g, settings, info, output.ptr, order,
					lib.enums.EquilSinkhorn)

				if info.converged:
					primal_feas = A.dot(output.x) - output.y
					dual_feas = A.T.dot(output.nu) + output.mu
					self.assertTrue(all(primal_feas <= 10 * settings.reltol))
					self.assertTrue(all(dual_feas <= 10 * settings.reltol))


			pxlib.function_vector_free(f)
			pxlib.function_vector_free(g)
			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_pogs_warmstart(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			pxlib = self.prox_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			lib = self.pogs_libs.get(dlib, pxlib,
									 single_precision=single_precision,
									 gpu=gpu)

			if lib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)
			pytype = np.float32 if single_precision else np.float64

			x_rand = np.random.rand(n).astype(pytype)
			nu_rand = np.random.rand(m).astype(pytype)

			f = pxlib.function_vector(0, None)
			g = pxlib.function_vector(0, None)

			pxlib.function_vector_calloc(f, m)
			pxlib.function_vector_calloc(g, n)
			f_py = np.zeros(m, dtype=pxlib.function)
			g_py = np.zeros(n, dtype=pxlib.function)
			f_ptr = f_py.ctypes.data_as(pxlib.function_p)
			g_ptr = g_py.ctypes.data_as(pxlib.function_p)
			for i in xrange(m):
				f_py[i] = pxlib.function(pxlib.enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = pxlib.function(pxlib.enums.IndGe0, 1, 0, 1, 0, 0)

			pxlib.function_vector_memcpy_va(f, f_ptr)
			pxlib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), dtype=pytype, order=pyorder)
				A += self.A_test
				A_ptr = A.ctypes.data_as(dlib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = PogsOutputLocal(dlib, lib, m, n, pytype)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)
				settings.maxiter = 0
				settings.x0 = x_rand.ctypes.data_as(dlib.ok_float_p)
				settings.nu0 = nu_rand.ctypes.data_as(dlib.ok_float_p)
				settings.warmstart = 1

				print "\nwarm start variable loading test (0 iters)"
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				if lib.full_api_accessible:
					# test warmstart feed

					rho = solver.contents.rho
					localA = np.zeros((m, n), dtype=pytype, order=pyorder)
					localA_ptr = localA.ctypes.data_as(dlib.ok_float_p)
					dlib.matrix_memcpy_am(localA_ptr,
										  solver.contents.M.contents.A, order)
					localvars = PogsVariablesLocal(m, n, pytype)
					self.load_all_local(dlib, localvars, solver)
					self.assertTrue(np.allclose(x_rand,
												localvars.e * localvars.x))
					self.assertTrue(np.allclose(nu_rand * -1/rho,
												localvars.d * localvars.yt))
					self.assertTrue(np.allclose(localA.dot(localvars.x),
												localvars.y))
					self.assertTrue(np.allclose(localA.T.dot(localvars.yt),
												-localvars.xt))

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
				settings.x0 = output.x.ctypes.data_as(dlib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# WARM START: x0, rho
				print "\nwarm start x0, rho"
				settings.resume = 0
				settings.rho = info.rho
				settings.x0 = output.x.ctypes.data_as(dlib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# WARM START: x0, nu0
				print "\nwarm start x0, nu0"
				settings.resume = 0
				settings.rho = 1
				settings.x0 = output.x.ctypes.data_as(dlib.ok_float_p)
				settings.nu0 = output.nu.ctypes.data_as(dlib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				# WARM START: x0, nu0
				print "\nwarm start x0, nu0, rho"
				settings.resume = 0
				settings.rho = info.rho
				settings.x0 = output.x.ctypes.data_as(dlib.ok_float_p)
				settings.nu0 = output.nu.ctypes.data_as(dlib.ok_float_p)
				settings.warmstart = 1
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				self.assertEqual(info.err, 0)
				self.assertTrue(info.converged or info.k >= settings.maxiter)

				lib.pogs_finish(solver, 0)


			pxlib.function_vector_free(f)
			pxlib.function_vector_free(g)
			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)

	def test_pogs_io(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			pxlib = self.prox_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			lib = self.pogs_libs.get(dlib, pxlib,
									 single_precision=single_precision,
									 gpu=gpu)

			if lib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)
			pytype = np.float32 if single_precision else np.float64

			x_rand = np.random.rand(n).astype(pytype)
			nu_rand = np.random.rand(m).astype(pytype)

			f = pxlib.function_vector(0, None)
			g = pxlib.function_vector(0, None)

			pxlib.function_vector_calloc(f, m)
			pxlib.function_vector_calloc(g, n)
			f_py = np.zeros(m, dtype=pxlib.function)
			g_py = np.zeros(n, dtype=pxlib.function)
			f_ptr = f_py.ctypes.data_as(pxlib.function_p)
			g_ptr = g_py.ctypes.data_as(pxlib.function_p)
			for i in xrange(m):
				f_py[i] = pxlib.function(pxlib.enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = pxlib.function(pxlib.enums.IndGe0, 1, 0, 1, 0, 0)

			pxlib.function_vector_memcpy_va(f, f_ptr)
			pxlib.function_vector_memcpy_va(g, g_ptr)


			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A = np.zeros((m, n), dtype=pytype, order=pyorder)
				A += self.A_test
				A_ptr = A.ctypes.data_as(dlib.ok_float_p)
				m, n = A.shape
				solver = lib.pogs_init(A_ptr, m, n, order,
									   lib.enums.EquilSinkhorn)

				output = PogsOutputLocal(dlib, lib, m, n, pytype)

				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan, np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											 None, None)
				lib.set_default_settings(settings)
				settings.verbose = 1

				print 'initial solve -> export data'
				lib.pogs_solve(solver, f, g, settings, info, output.ptr)
				k_orig = info.k

				mindim = min(m, n)
				A_equil = np.zeros((m, n), order=pyorder).astype(pytype)
				A_equil_ptr = A_equil.ctypes.data_as(dlib.ok_float_p)
				if lib.direct:
					LLT = np.zeros((mindim, mindim), order=pyorder)
					LLT = LLT.astype(pytype)
					LLT_ptr = LLT.ctypes.data_as(dlib.ok_float_p)
				else:
					LLT = c_void_p()
					LLT_ptr = LLT

				d = np.zeros(m).astype(pytype)
				e = np.zeros(n).astype(pytype)
				z = np.zeros(m + n).astype(pytype)
				z12 = np.zeros(m + n).astype(pytype)
				zt = np.zeros(m + n).astype(pytype)
				zt12 = np.zeros(m + n).astype(pytype)
				zprev = np.zeros(m + n).astype(pytype)
				rho = np.zeros(1).astype(pytype)

				d_ptr =d.ctypes.data_as(dlib.ok_float_p)
				e_ptr = e.ctypes.data_as(dlib.ok_float_p)
				z_ptr = z.ctypes.data_as(dlib.ok_float_p)
				z12_ptr = z12.ctypes.data_as(dlib.ok_float_p)
				zt_ptr = zt.ctypes.data_as(dlib.ok_float_p)
				zt12_ptr = zt12.ctypes.data_as(dlib.ok_float_p)
				zprev_ptr = zprev.ctypes.data_as(dlib.ok_float_p)
				rho_ptr = rho.ctypes.data_as(dlib.ok_float_p)

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

			pxlib.function_vector_free(f)
			pxlib.function_vector_free(g)
			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)