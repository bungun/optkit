import unittest
import os
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from ctypes import c_void_p, byref, cast, addressof
from optkit.libs import PogsAbstractLibs
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
from optkit.tests.C.pogs_helper import PogsVariablesLocal, PogsOutputLocal
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

class PogsAbstractLibsTestCase(unittest.TestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = PogsAbstractLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue(any(libs))

class PogsAbstractTestCases(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = PogsAbstractLibs()

		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.A_test_sparse = self.A_test
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)
			self.A_test_sparse = np.zeros(self.shape)
			self.A_test_sparse += self.A_test
			for i in xrange(self.shape[0]):
				if np.random.rand() > 0.4:
					self.A_test_sparse[i, :] *= 0
			for j in xrange(self.shape[1]):
				if np.random.rand() > 0.4:
					self.A_test_sparse[:, j] *= 0

		self.nnz = sum(sum(self.A_test_sparse > 0))

		self.x_test = np.random.rand(self.shape[1])

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	@staticmethod
	def load_to_local(lib, py_vector, c_vector):
		lib.vector_memcpy_av(
				py_vector.ctypes.data_as(lib.ok_float_p), c_vector, 1)

	def load_all_local(self, lib, py_vars, solver):
		if not isinstance(py_vars, PogsVariablesLocal):
			raise TypeError('argument "py_vars" must be of type {}'.format(
							PogsVariablesLocal))

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
		self.load_to_local(lib, py_vars.d, solver.contents.W.contents.d)
		self.load_to_local(lib, py_vars.e, solver.contents.W.contents.e)

	def pogs_equilibration(self, lib, solver, A, opA, localvars):
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 2 * 10**(-DIGITS)
		ATOLM = RTOL * A.shape[0]**0.5

		m, n = A.shape
		d_local = np.zeros(m).astype(lib.pyfloat)
		e_local = np.zeros(n).astype(lib.pyfloat)
		self.load_to_local(lib, d_local, solver.contents.W.contents.d)
		self.load_to_local(lib, e_local, solver.contents.W.contents.e)

		x_rand = np.random.rand(n).astype(lib.pyfloat)
		x_ptr = x_rand.ctypes.data_as(lib.ok_float_p)
		y_out = np.zeros(m).astype(lib.pyfloat)
		y_ptr = y_out.ctypes.data_as(lib.ok_float_p)

		x = lib.vector(0, 0, None)
		y = lib.vector(0, 0, None)

		lib.vector_calloc(x, n);
		lib.vector_calloc(y, m);
		self.register_var('x', x, lib.vector_free)
		self.register_var('y', y, lib.vector_free)

		lib.vector_memcpy_va(x, x_ptr, 1)
		opA.contents.apply(opA.contents.data, x, y)
		lib.vector_memcpy_av(y_ptr, y, 1)

		A_eqx = y_out
		DAEx = d_local * A.dot(e_local * x_rand)

		self.assertTrue(np.linalg.norm(A_eqx - DAEx) <=
						ATOLM + RTOL * np.linalg.norm(DAEx))

		self.free_var('x')
		self.free_var('y')

	def pogs_projector(self, lib, blas_handle, solver, opA):
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * opA.contents.size1**0.5

		m, n = (opA.contents.size1, opA.contents.size2)

		x_in = lib.vector(0, 0, None)
		y_in = lib.vector(0, 0, None)
		x_out = lib.vector(0, 0, None)
		y_out = lib.vector(0, 0, None)
		y_c = lib.vector(0, 0, None)

		lib.vector_calloc(x_in, n)
		lib.vector_calloc(y_in, m)
		lib.vector_calloc(x_out, n)
		lib.vector_calloc(y_out, m)
		lib.vector_calloc(y_c, m)
		self.register_var('x_in', x_in, lib.vector_free)
		self.register_var('y_in', y_in, lib.vector_free)
		self.register_var('x_out', x_out, lib.vector_free)
		self.register_var('y_out', y_out, lib.vector_free)
		self.register_var('y_c', y_c, lib.vector_free)

		x_in_py = np.random.rand(n).astype(lib.pyfloat)
		y_in_py = np.random.rand(m).astype(lib.pyfloat)
		x_out_py = np.zeros(n).astype(lib.pyfloat)
		y_out_py = np.zeros(m).astype(lib.pyfloat)
		y_ = np.zeros(m).astype(lib.pyfloat)

		x_in_ptr = x_in_py.ctypes.data_as(lib.ok_float_p)
		y_in_ptr = y_in_py.ctypes.data_as(lib.ok_float_p)
		y_ptr = y_.ctypes.data_as(lib.ok_float_p)

		lib.vector_memcpy_va(x_in, x_in_ptr, 1)
		lib.vector_memcpy_va(y_in, y_in_ptr, 1)

		# project (x_in, y_in) onto {(x, y) | y=Ax} to obtain (x_out, y_out)
		P = solver.contents.W.contents.P
		P.contents.project.argtypes
		P.contents.project(P.contents.data, x_in, y_in, x_out, y_out,
						   1e-3 * RTOL)

		self.load_to_local(lib, x_out_py, x_out)
		self.load_to_local(lib, y_out_py, y_out)

		# form y_ = Ax_out
		opA.contents.apply(opA.contents.data, x_out, y_c)
		self.load_to_local(lib, y_, y_c)

		# test Ax_out ~= y_out
		self.assertTrue(np.linalg.norm(y_ - y_out_py) <=
						ATOLM + RTOL * np.linalg.norm(y_))

		self.free_var('x_in')
		self.free_var('y_in')
		self.free_var('x_out')
		self.free_var('y_out')
		self.free_var('y_c')

	def pogs_scaling(self, lib, solver, f, f_py, g, g_py, localvars):

		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU

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
		err = lib.update_problem(solver, f, g)
		self.assertEqual(err, 0)

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

	def pogs_warmstart(self, lib, solver, settings, localA, localvars):
		self.assertNotEqual(solver, None)
		self.assertEqual(type(solver), lib.pogs_solver_p)

		m, n = localA.shape
		DIGITS = 7 - 2*lib.FLOAT - 1*lib.GPU
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

		rho = solver.contents.rho

		x_rand = np.random.rand(n).astype(lib.pyfloat)
		nu_rand = np.random.rand(m).astype(lib.pyfloat)

		x_ptr = x_rand.ctypes.data_as(lib.ok_float_p)
		nu_ptr = nu_rand.ctypes.data_as(lib.ok_float_p)

		settings.x0 = x_ptr
		settings.nu0 = nu_ptr
		err = lib.update_settings(solver.contents.settings, settings)
		self.assertEqual(err, 0)

		err = lib.initialize_variables(solver)
		self.assertEqual(err, 0)

		self.load_all_local(lib, localvars, solver)


		self.assertTrue(np.linalg.norm(x_rand - localvars.e * localvars.x) <=
						ATOLN + RTOL * np.linalg.norm(x_rand))
		self.assertTrue(
				np.linalg.norm(nu_rand * localvars.d + rho * localvars.yt) <=
				ATOLM + RTOL * np.linalg.norm(rho * localvars.yt))
		self.assertTrue(
				np.linalg.norm(localA.dot(localvars.x) - localvars.y) <=
				ATOLM + RTOL * np.linalg.norm(localvars.y))
		self.assertTrue(
				np.linalg.norm(localA.T.dot(localvars.yt) + localvars.xt) <=
				ATOLN + RTOL * np.linalg.norm(localvars.xt))

	def pogs_primal_update(self, lib, solver, localvars):
		"""primal update test

			set

				z^k = z^{k-1}

			check

				z^k == z^{k-1}

			holds elementwise
		"""
		self.assertNotEqual(solver, None)
		self.assertEqual(type(solver), lib.pogs_solver_p)

		m, n = localvars.m, localvars.n
		RTOL = 1e-8
		ATOLMN = RTOL * (m * n)**0.5

		err = lib.set_prev(solver.contents.z)
		self.assertEqual(err, 0)

		self.load_all_local(lib, localvars, solver)
		self.assertTrue(np.linalg.norm(localvars.z - localvars.prev) <=
						ATOLMN + RTOL * np.linalg.norm(localvars.prev))

	def pogs_prox(self, lib, blas_handle, solver, f, f_py, g, g_py, localvars):
		"""proximal operator application test

			set

				x^{k+1/2} = prox_{g, rho}(x^k - xt^k)
				y^{k+1/2} = prox_{f, rho}(y^k - yt^k)

			in C and Python, check that results agree
		"""
		self.assertNotEqual(solver, None)
		self.assertEqual(type(solver), lib.pogs_solver_p)

		m, n = localvars.m, localvars.n
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * (m)**0.5
		ATOLN = RTOL * (n)**0.5

		err = lib.prox(blas_handle, f, g, solver.contents.z,
						 solver.contents.rho)
		self.assertEqual(err, 0)

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
		self.assertTrue(np.linalg.norm(localvars.x12 - x_out) <=
						ATOLN + RTOL * np.linalg.norm(x_out))
		self.assertTrue(np.linalg.norm(localvars.y12 - y_out) <=
						ATOLM + RTOL * np.linalg.norm(y_out))

	def pogs_primal_project(self, lib, blas_handle, solver, settings, opA,
							localvars):
		"""primal projection test

			set

				(x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})

			check that


				y^{k+1/2} == A * x^{k+1/2}

			holds to numerical tolerance
		"""
		self.assertNotEqual(solver, None)
		self.assertEqual(type(solver), lib.pogs_solver_p)

		m, n = localvars.m, localvars.n
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 2 * 10**(-DIGITS)
		ATOLM = RTOL * (m)**0.5

		err = lib.project_primal(blas_handle, solver.contents.W.contents.P,
							   solver.contents.z, settings.alpha)
		self.assertEqual(err, 0)


		self.load_all_local(lib, localvars, solver)

		y = lib.vector(0, 0, None)
		lib.vector_calloc(y, opA.contents.size1)
		self.register_var('y', y, lib.vector_free)
		y_ = np.zeros(m).astype(lib.pyfloat)

		# form y_ = Ax_out
		x = solver.contents.z.contents.primal.contents.x
		opA.contents.apply(opA.contents.data, x, y)
		self.load_to_local(lib, y_, y)

		self.free_var('y')

		self.assertTrue(np.linalg.norm(y_ - localvars.y) <=
						ATOLM + RTOL * np.linalg.norm(localvars.y))

	def pogs_dual_update(self, lib, blas_handle, solver, settings, localvars):
		"""dual update test

			set

				zt^{k+1/2} = z^{k+1/2} - z^{k} + zt^{k}
				zt^{k+1} = zt^{k} - z^{k+1} + alpha * z^{k+1/2} +
											  (1-alpha) * z^k

			in C and Python, check that results agree
		"""
		self.assertNotEqual(solver, None)
		self.assertEqual(type(solver), lib.pogs_solver_p)

		m, n = localvars.m, localvars.n
		DIGITS = 5 if lib.FLOAT else 7
		RTOL = 10**(-DIGITS)
		ATOLMN = RTOL * (m + n)**0.5

		self.load_all_local(lib, localvars, solver)
		alpha = settings.alpha
		zt12_py = localvars.z12 - localvars.prev + localvars.zt
		zt_py = localvars.zt - localvars.z + (alpha * localvars.z12 +
								 			   (1-alpha) * localvars.prev)

		err = lib.update_dual(blas_handle, solver.contents.z, alpha)
		self.assertEqual(err, 0)

		self.load_all_local(lib, localvars, solver)
		self.assertTrue(np.linalg.norm(localvars.zt12 - zt12_py) <=
						ATOLMN + RTOL * np.linalg.norm(zt12_py))
		self.assertTrue(np.linalg.norm(localvars.zt - zt_py) <=
						ATOLMN + RTOL * np.linalg.norm(zt_py))

	def pogs_check_convergence(self, lib, blas_handle, solver, f_list, g_list,
							   objectives, residuals, tolerances, settings,
							   opA, A, localvars):
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
		m, n = localvars.m, localvars.n
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

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

		self.assertTrue(abs(objectives.gap - obj_gap_py) <= RTOL*obj_gap_py)
		self.assertTrue(abs(tolerances.primal - tol_primal) <= RTOL*tol_primal)
		self.assertTrue(abs(tolerances.dual - tol_dual) <= RTOL*tol_dual)

		A_x12 = lib.vector(0, 0, None)
		At_yt12 = lib.vector(0, 0, None)
		lib.vector_calloc(A_x12, m)
		lib.vector_calloc(At_yt12, n)
		self.register_var('Ax', A_x12, lib.vector_free)
		self.register_var('Ay', At_yt12, lib.vector_free)

		opA.contents.adjoint(opA.contents.data,
			solver.contents.z.contents.dual12.contents.y, At_yt12)
		opA.contents.apply(opA.contents.data,
			solver.contents.z.contents.primal12.contents.x, A_x12)

		A_x12_ = np.zeros(m).astype(lib.pyfloat)
		At_yt12_ = np.zeros(n).astype(lib.pyfloat)
		self.load_to_local(lib, A_x12_, A_x12)
		self.load_to_local(lib, At_yt12_, At_yt12)

		self.free('Ax')
		self.free('Ay')

		res_primal = np.linalg.norm(A_x12_ - localvars.y12)
		res_dual = np.linalg.norm(At_yt12_ + localvars.xt12)

		self.assertTrue(abs(residuals.primal - res_primal) <= RTOL*res_primal)
		self.assertTrue(abs(residuals.dual - res_dual) <= RTOL*res_dual)
		self.assertTrue(abs(residuals.gap - abs(obj_gap_py)) <=
						RTOL*obj_gap_py)

		converged_py = res_primal <= tolerances.primal and \
					   res_dual <= tolerances.dual

		self.assertEqual(converged, converged_py)

	def pogs_adapt_rho(self, lib, solver, settings, residuals, tolerances,
					   localvars):
		"""adaptive rho test

			rho is rescaled
			dual variable is rescaled accordingly

			the following equality should hold elementwise:

				rho_before * zt_before == rho_after * zt_after

			(here zt, or z tilde, is the dual variable)
		"""
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU

		if settings.adaptiverho:
			rho_params = lib.adapt_params(1.05, 0, 0, 1)
			zt_before = localvars.zt
			rho_before = solver.contents.rho
			err = lib.adaptrho(solver, rho_params, residuals, tolerances,
								   1)
			self.assertEqual(err, 0)
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
		DIGITS = 3 - 1 * lib.FLOAT

		if not isinstance(solver, lib.pogs_solver_p):
			raise TypeError('argument "solver" must be of type {}'.format(
							lib.pogs_solver_p))

		if not isinstance(output, PogsOutputLocal):
			raise TypeError('argument "output" must be of type {}'.format(
							PogsOutputLocal))

		if not isinstance(localvars, PogsVariablesLocal):
			raise TypeError('argument "localvars" must be of type {}'.format(
							PogsVariablesLocal))

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


	@property
	def optypes(self):
	    return ['dense', 'sparse']

	def gen_prox(self, lib):
		m, n = self.shape

		f = lib.function_vector(0, None)
		g = lib.function_vector(0, None)
		lib.function_vector_calloc(f, m)
		lib.function_vector_calloc(g, n)
		f_py = np.zeros(m).astype(lib.function)
		g_py = np.zeros(n).astype(lib.function)
		f_ptr = f_py.ctypes.data_as(lib.function_p)
		g_ptr = g_py.ctypes.data_as(lib.function_p)

		for i in xrange(m):
			f_py[i] = lib.function(lib.enums.Abs, 1, 1, 1, 0, 0)

		for j in xrange(n):
			g_py[j] = lib.function(lib.enums.IndGe0, 1, 0, 1, 0, 0)

		lib.function_vector_memcpy_va(f, f_ptr)
		lib.function_vector_memcpy_va(g, g_ptr)

		return f, g, f_py, g_py

	def gen_pogs_operator(self, lib, type_='dense'):
		m, n = self.shape

		if type_ not in self.optypes:
			raise ValueError('argument "type" must be one of {}'.format(
							 self.optypes))

		if type_ == 'dense':
			A_ = self.A_test.astype(lib.pyfloat)
			A_ptr = A_.ctypes.data_as(lib.ok_float_p)
			order = lib.enums.CblasRowMajor if A_.flags.c_contiguous \
					else lib.enums.CblasColMajor
			o = lib.pogs_dense_operator_gen(A_ptr, m, n, order),
				   lib.pogs_dense_operator_free
			free_o = lib.pogs_dense_operator_free
		elif type_ == 'sparse':
			A_ = self.A_test_sparse
			A_sp = csr_matrix(A_.astype(lib.pyfloat))
			A_ptr = A_sp.indptr.ctypes.data_as(lib.ok_int_p)
			A_ind = A_sp.indices.ctypes.data_as(lib.ok_int_p)
			A_val = A_sp.data.ctypes.data_as(lib.ok_float_p)
			order = lib.enums.CblasRowMajor
			o = lib.pogs_sparse_operator_gen(A_val, A_ind, A_ptr, m, n,
												 self.nnz, order)
			free_o = lib.pogs_sparse_operator_free
		else:
			raise RuntimeError('this should be unreachable due to ValueError '
							   'raised above')

		return A_, o, free_o

	def test_default_settings(self)
		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None,
										 None)

			lib.set_default_settings(settings)
			TOL = 1e-4
			self.assertTrue(abs(settings.alpha - ALPHA_DEFAULT) <=
							TOL * ALPHA_DEFAULT)
			self.assertTrue(abs(settings.rho - RHO_DEFAULT) <=
							TOL * RHO_DEFAULT)
			self.assertTrue(abs(settings.abstol - ABSTOL_DEFAULT) <=
							TOL * ABSTOL_DEFAULT)
			self.assertTrue(abs(settings.reltol - RELTOL_DEFAULT) <=
							TOL * RELTOL_DEFAULT)
			self.assertTrue(abs(settings.maxiter - MAXITER_DEFAULT) <=
							TOL * MAXITER_DEFAULT)
			self.assertTrue(abs(settings.verbose - VERBOSE_DEFAULT) <=
							TOL * VERBOSE_DEFAULT)
			self.assertTrue(abs(settings.suppress - SUPPRESS_DEFAULT) <=
							TOL * SUPPRESS_DEFAULT)
			self.assertTrue(abs(settings.adaptiverho - ADAPTIVE_DEFAULT) <=
							TOL * ADAPTIVE_DEFAULT)
			self.assertTrue(abs(settings.gapstop - GAPSTOP_DEFAULT) <=
							TOL * GAPSTOP_DEFAULT)
			self.assertTrue(abs(settings.warmstart - WARMSTART_DEFAULT) <=
							TOL * WARMSTART_DEFAULT)
			self.assertTrue(abs(settings.resume - RESUME_DEFAULT) <=
							TOL * RESUME_DEFAULT)

	def test_operator_gen_free(self)
		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for optype in self.optypes:
				_, o, free_o = self.gen_pogs_operator(lib, optype)
				self.register_var('o', o, free_o)
				self.assertEqual(type(o), olib.operator_p)
				self.free_var('o')

	def test_pogs_init_finish(self)
		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			NO_DEVICE_RESET = 0

			for optype in self.optypes:
				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:
						_, o, free_o = self.gen_pogs_operator(lib, optype)
						self.register_var('o', o, free_o)
						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.assertNotEqual(solver, 0)
						lib.pogs_finish(solver, NO_DEVICE_RESET)
						self.del_pogs_operator(lib, o, optype)

				# now reset device (once operator is freed)
				self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_private_api(self):
		hdl = c_void_p()
		m, n = self.shape
		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			NO_DEVICE_RESET = 0
			for optype in self.optypes:
				self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

				f, g, f_py, g_py = self.gen_prox(lib)
				self.register_var('f', f, lib.function_vector_free)
				self.register_var('g', g, lib.function_vector_free)

				f_list = [lib.function(*f_) for f_ in f_py]
				g_list = [lib.function(*g_) for g_ in g_py]

				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:

						A, o, free_o = self.gen_pogs_operator(lib, optype)
						self.register_var('o', o, free_o)

						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.assertNotEqual(solver, 0)

						output = PogsOutputLocal(lib, lib, m, n)
						info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan,
											 np.nan)
						settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0,
													 0, 0, None, None)
						lib.set_default_settings(settings)

						localvars = PogsVariablesLocal(m, n, lib.pyfloat)

						if lib.full_api_accessible:
							slvr = cast(solver, lib.pogs_solver_p)
							res = lib.pogs_residuals(0, 0, 0)
							tols = lib.make_tolerances(settings, m, n)
							obj = lib.pogs_objectives(0, 0, 0)
							zz = slvr.contents.z.contents

							self.pogs_equilibration(lib, slvr, A, o,
													localvars)
							self.pogs_projector(lib, lib, hdl, slvr, o)
							self.pogs_scaling(lib, lib, lib, slvr, f,
											  f_py, g, g_py, localvars)
							self.pogs_primal_update(lib, lib, slvr,
													localvars)
							self.pogs_prox(lib, lib, lib, hdl, slvr, f,
										   f_py, g, g_py, localvars)
							self.pogs_primal_project(lib, lib, hdl, slvr,
													 settings, o, localvars)
							self.pogs_dual_update(lib, lib, hdl, slvr,
												  settings, localvars)
							self.pogs_check_convergence(lib, lib, hdl, slvr,
														f_list, g_list, obj,
														res, tols, settings,
														o, A, localvars)
							self.pogs_adapt_rho(lib, lib, slvr, settings, res,
												tols, localvars)
							self.pogs_unscaling(lib, lib, slvr, output,
												localvars)

						lib.pogs_finish(solver, NO_DEVICE_RESET)
						self.free_var('o')

				self.free_var('f')
				self.free_var('g')

				# now reset device (once operator is freed)
				self.assertEqual(lib.blas_destroy_handle(hdl), 0)
				self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_call(self):
		hdl = c_void_p()
		m, n = self.shape
		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			NO_DEVICE_RESET = 0
			for optype in self.optypes:
				self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

				f, g, f_py, g_py = self.gen_prox(lib)
				self.register_var('f', f, lib.function_vector_free)
				self.register_var('g', g, lib.function_vector_free)

				f_list = [lib.function(*f_) for f_ in f_py]
				g_list = [lib.function(*g_) for g_ in g_py]

				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:
						A, o, free_o = self.gen_pogs_operator(lib, optype)
						self.register_var('o', o, free_o)

						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.assertNotEqual(solver, 0)

						output = PogsOutputLocal(lib, lib, m, n)
						info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan,
											 np.nan)
						settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0,
													 0, 0, None, None)
						lib.set_default_settings(settings)
						settings.verbose = 0

						localvars = PogsVariablesLocal(m, n, lib.pyfloat)

						err = lib.pogs_solve(solver, f, g, settings, info,
									   output.ptr)
						self.assertEqual(err, 0)

						err = lib.pogs_finish(solver, NO_DEVICE_RESET)
						self.assertEqual(err, 0)

						self.free_var('o')

						if info.converged:
							rtol = settings.reltol
							atolm = settings.abstol * (m**0.5)
							atoln = settings.abstol * (n**0.5)
							y_norm = np.linalg.norm(output.y)
							mu_norm = np.linalg.norm(output.mu)

							primal_feas = np.linalg.norm(
									A.dot(output.x) - output.y)
							dual_feas = np.linalg.norm(
									A.T.dot(output.nu) + output.mu)

							self.assertTrue(primal_feas <=
											10 * (atolm + rtol * y_norm))
							self.assertTrue(dual_feas <=
											20 * (atoln + rtol * mu_norm))

				self.free_var('f')
				self.free_var('g')
				self.assertEqual(lib.blas_destroy_handle(hdl), 0)
				self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_call_unified(self):
		hdl = c_void_p()
		m, n = self.shape
		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			NO_DEVICE_RESET = 0
			for optype in self.optypes:
				self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

				f, g, f_py, g_py = self.gen_prox(lib)
				self.register_var('f', f, lib.function_vector_free)
				self.register_var('g', g, lib.function_vector_free)
				f_list = [lib.function(*f_) for f_ in f_py]
				g_list = [lib.function(*g_) for g_ in g_py]

				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:
						A, o, free_o = self.gen_pogs_operator(lib, optype)
						self.register_var('o', o, free_o)

						output = PogsOutputLocal(lib, lib, m, n)
						info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan,
											 np.nan)
						settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0,
													 0, 0, None, None)
						lib.set_default_settings(settings)
						localvars = PogsVariablesLocal(m, n, lib.pyfloat)

						err = lib.pogs(o, f, g, settings, info, output.ptr,
									   DIRECT, EQUILNORM, NO_DEVICE_RESET)

						self.free_var('o')
						self.assertEqual(err, 0)

						if info.converged:
							rtol = settings.reltol
							atolm = settings.abstol * (m**0.5)
							atoln = settings.abstol * (n**0.5)
							y_norm = np.linalg.norm(output.y)
							mu_norm = np.linalg.norm(output.mu)

							primal_feas = np.linalg.norm(
									A.dot(output.x) - output.y)
							dual_feas = np.linalg.norm(
									A.T.dot(output.nu) + output.mu)

							self.assertTrue(primal_feas <=
											10 * (atolm + rtol * y_norm))
							self.assertTrue(dual_feas <=
											20 * (atoln + rtol * mu_norm))

				self.free_var('f')
				self.free_var('g')
				self.assertEqual(lib.blas_destroy_handle(hdl), 0)
				self.assertEqual(lib.ok_device_reset(), 0)

	def test_pogs_warmstart(self):
		hdl = c_void_p()
		m, n = self.shape
		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			NO_DEVICE_RESET = 0
			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5
			ATOLM = RTOL * m**0.5

			for optype in self.optypes:
				self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

				f, g, f_py, g_py = self.gen_prox(lib)
				self.register_var('f', f, lib.function_vector_free)
				self.register_var('g', g, lib.function_vector_free)
				f_list = [lib.function(*f_) for f_ in f_py]
				g_list = [lib.function(*g_) for g_ in g_py]

				for DIRECT in [0, 1]:
					EQUILNORM = 1.

				x_rand = np.random.rand(n).astype(lib.pyfloat)
				nu_rand = np.random.rand(m).astype(lib.pyfloat)

				A, o, free_o = self.gen_pogs_operator(lib, lib, optype)
				self.register_var('o', o, free_o)

				solver = lib.pogs_init(o, DIRECT, EQUILNORM)
				self.assertNotEqual(solver, 0)

				output = PogsOutputLocal(lib, lib, m, n)
				info = lib.pogs_info(0, 0, 0, np.nan, np.nan, np.nan,
									 np.nan)
				settings = lib.pogs_settings(0, 0, 0, 0, 0, 0, 0, 0, 0,
											 0, 0, None, None)

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
					localvars = PogsVariablesLocal(m, n, lib.pyfloat)
					slvr = cast(solver, lib.pogs_solver_p)
					rho = slvr.contents.rho
					self.load_all_local(lib, localvars, slvr)
					self.assertTrue(np.linalg.norm(
							x_rand - localvars.e * localvars.x) <=
							ATOLN + RTOL * np.linalg.norm(x_rand))
					self.assertTrue(np.linalg.norm(
							nu_rand + rho * localvars.d * localvars.yt) <=
							ATOLM + RTOL * np.linalg.norm(nu_rand))
					self.assertTrue(np.linalg.norm(
							localvars.d * A.dot(localvars.e * localvars.x) -
							localvars.y) <=
							ATOLM + RTOL * np.linalg.norm(localvars.y))
					self.assertTrue(np.linalg.norm(
							localvars.e * A.T.dot(localvars.d * localvars.yt) +
							localvars.xt) <=
							ATOLN + RTOL * np.linalg.norm(localvars.xt))

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

				self.assertEqual(lib.pogs_finish(solver, 0), 0)
				self.free_var('o')

			self.free_var('f')
			self.free_var('g')
			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)

"""
	def test_pogs_io(self):
		hdl = c_void_p()
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			lib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			lib = self.prox_libs.get(
					lib, single_precision=single_precision, gpu=gpu)
			lib = self.libs.get(
					lib, lib, single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

			x_rand = np.random.rand(n).astype(lib.pyfloat)
			nu_rand = np.random.rand(m).astype(lib.pyfloat)

			f = lib.function_vector(0, None)
			g = lib.function_vector(0, None)

			lib.function_vector_calloc(f, m)
			lib.function_vector_calloc(g, n)
			f_py = np.zeros(m).astype(lib.function)
			g_py = np.zeros(n).astype(lib.function)
			f_ptr = f_py.ctypes.data_as(lib.function_p)
			g_ptr = g_py.ctypes.data_as(lib.function_p)
			for i in xrange(m):
				f_py[i] = lib.function(lib.enums.Abs, 1, 1, 1, 0, 0)

			for j in xrange(n):
				g_py[j] = lib.function(lib.enums.IndGe0, 1, 0, 1, 0, 0)

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

				output = PogsOutputLocal(lib, lib, m, n)

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

			lib.function_vector_free(f)
			lib.function_vector_free(g)
			self.assertEqual(lib.blas_destroy_handle(hdl), 0)
			self.assertEqual(lib.ok_device_reset(), 0)
"""