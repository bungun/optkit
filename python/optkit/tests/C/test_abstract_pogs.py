import os
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from ctypes import c_void_p, byref, cast, addressof
from optkit.utils.proxutils import func_eval_python
from optkit.libs.pogs import PogsAbstractLibs
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.base import OptkitCOperatorTestCase
from optkit.tests.C.pogs_base import OptkitCPogsTestCase

class PogsAbstractLibsTestCase(OptkitTestCase):
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
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue( any(libs) )

class PogsAbstractTestCases(OptkitCPogsTestCase, OptkitCOperatorTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = PogsAbstractLibs()
		self.A_test = self.A_test_gen
		self.A_test_sparse = self.A_test_sparse_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def register_pogs_operator(self, lib, type_='dense', name='o'):
		m, n = self.shape

		if type_ not in self.op_keys:
			raise ValueError('argument "type" must be one of {}'.format(
							 self.op_keys))

		if type_ == 'dense':
			A_ = self.A_test.astype(lib.pyfloat)
			A_ptr = A_.ctypes.data_as(lib.ok_float_p)
			order = lib.enums.CblasRowMajor if A_.flags.c_contiguous \
					else lib.enums.CblasColMajor
			o = lib.pogs_dense_operator_gen(A_ptr, m, n, order)
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

		self.register_var(name, o.contents.data, o.contents.free)
		return A_, o

	def assert_pogs_equilibration(self, lib, solver, A, opA, local_vars):
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 2 * 10**(-DIGITS)
		ATOLM = RTOL * A.shape[0]**0.5

		m, n = A.shape
		d_local = np.zeros(m).astype(lib.pyfloat)
		e_local = np.zeros(n).astype(lib.pyfloat)
		self.load_to_local(lib, d_local, solver.contents.W.contents.d)
		self.load_to_local(lib, e_local, solver.contents.W.contents.e)

		x, x_rand, x_ptr = self.register_vector(lib, n, 'x')
		y, y_out, y_ptr = self.register_vector(lib, m, 'y')

		self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )
		self.assertCall( opA.contents.apply(opA.contents.data, x, y) )
		self.assertCall( lib.vector_memcpy_av(y_ptr, y, 1) )

		A_eqx = y_out
		DAEx = d_local * A.dot(e_local * x_rand)

		self.assertVecEqual( A_eqx, DAEx, ATOLM, RTOL)
		self.free_vars('x', 'y')

	def assert_pogs_projector(self, lib, blas_handle, projector, opA):
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * opA.contents.size1**0.5

		m, n = (opA.contents.size1, opA.contents.size2)

		x_in, x_in_py, x_in_ptr = self.register_vector(lib, n, 'x_in')
		y_in, y_in_py, y_in_ptr = self.register_vector(lib, m, 'y_in')
		x_out, x_out_py, x_out_ptr = self.register_vector(lib, n, 'x_out')
		y_out, y_out_py, y_out_ptr = self.register_vector(lib, m, 'y_out')
		y_c, y_, y_ptr = self.register_vector(lib, m, 'y_c')

		x_in_py += np.random.rand(n).astype(lib.pyfloat)
		y_in_py += np.random.rand(m).astype(lib.pyfloat)
		self.assertCall( lib.vector_memcpy_va(x_in, x_in_ptr, 1) )
		self.assertCall( lib.vector_memcpy_va(y_in, y_in_ptr, 1) )

		# project (x_in, y_in) onto {(x, y) | y=Ax} to obtain (x_out, y_out)
		self.assertCall( projector.contents.project(projector.contents.data,
													x_in, y_in, x_out, y_out,
													1e-3 * RTOL) )

		self.load_to_local(lib, x_out_py, x_out)
		self.load_to_local(lib, y_out_py, y_out)

		# form y_ = Ax_out
		self.assertCall( opA.contents.apply(opA.contents.data, x_out, y_c) )
		self.load_to_local(lib, y_, y_c)

		# test Ax_out ~= y_out
		self.assertVecEqual( y_, y_out_py, ATOLM, RTOL )

		self.free_vars('x_in', 'y_in', 'x_out', 'y_out', 'y_c')

	def assert_pogs_primal_project(self, lib, blas_handle, solver, local_vars):
		"""primal projection test
			set
				(x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})
			check that
				y^{k+1} == A * x^{k+1}
			holds to numerical tolerance
		"""
		m = solver.contents.z.contents.m
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5

		projector = solver.contents.W.contents.P
		z = solver.contents.z
		x = z.contents.primal.contents.x
		alpha = solver.contents.settings.contents.alpha
		operator_ = solver.contents.W.contents.A.contents

		self.assertCall( lib.project_primal( blas_handle, projector, z, alpha,
											 1e-3*RTOL) )


		Ax, Ax_py, Ax_ptr = self.register_vector(lib, m, 'Ax')
		self.assertCall( operator_.apply(operator_.data, x, Ax))
		self.load_to_local(lib, Ax_py, Ax)
		self.load_all_local(lib, local_vars, solver)
		self.assertVecEqual( Ax_py, local_vars.y, ATOLM, RTOL)

		self.free_var('Ax')

	def assert_pogs_warmstart(self, lib, solver, local_vars, x0, nu0):
		m, n = len(nu0), len(x0)
		DIGITS = 7 - 2 * lib.FLOAT
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * m**0.5
		ATOLN = RTOL * n**0.5

		z = solver.contents.z
		rho = solver.contents.rho
		opA = solver.contents.W.contents.A.contents
		self.load_all_local(lib, local_vars, solver)

		Ax, Ax_py, Ax_ptr = self.register_vector(lib, m, 'Ax')
		Atnu, Atnu_py, Atnu_ptr = self.register_vector(lib, n, 'Atnu')

		self.assertCall( opA.apply(
				opA.data, z.contents.primal.contents.x, Ax) )
		self.assertCall( opA.adjoint(
				opA.data, z.contents.dual.contents.y, Atnu) )

		self.load_to_local(lib, Ax_py, Ax)
		self.load_to_local(lib, Atnu_py, Atnu)

		# check variable state is consistent after warm start
		self.assertVecEqual(
				x0, local_vars.e * local_vars.x, ATOLN, RTOL )
		self.assertVecEqual(
				nu0 * -1./rho, local_vars.d * local_vars.yt,
				ATOLM, RTOL )
		self.assertVecEqual(
				Ax_py, local_vars.y, ATOLM, RTOL )
		self.assertVecEqual(
				Atnu_py, -local_vars.xt, ATOLN, RTOL )

		self.free_vars('Ax', 'Atnu')

	def assert_pogs_check_convergence(self, lib, blas_handle, solver, f_list,
									  g_list, objectives, residuals,
									  tolerances, local_vars):
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
		m, n = local_vars.m, local_vars.n
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

		z = solver.contents.z.contents
		opA = solver.contents.W.contents.A.contents
		dy, dy_py, dy_ptr = self.register_vector(lib, m, 'dy')
		dmu, dmu_py, dmu_ptr = self.register_vector(lib, n, 'dmu')

		self.assertCall( lib.vector_memcpy_vv(dy, z.primal12.contents.y) )
		self.assertCall( opA.fused_apply(opA.data, 1, z.primal12.contents.x,
										 -1, dy) )
		self.assertCall( lib.vector_memcpy_av(dy_ptr, dy, 1) )
		res_primal = np.linalg.norm(dy_py)

		self.assertCall( lib.vector_memcpy_vv(dmu, z.dual12.contents.x) )
		self.assertCall( opA.fused_adjoint(opA.data, 1, z.dual12.contents.y,
										 1, dmu) )
		self.assertCall( lib.vector_memcpy_av(dmu_ptr, dmu, 1) )
		res_dual = np.linalg.norm(dmu_py)

		self.assertScalarEqual( residuals.primal, res_primal, RTOL )
		self.assertScalarEqual( residuals.dual, res_dual, RTOL )
		self.assertScalarEqual( residuals.gap, abs(obj_gap_py), RTOL )

		converged_py = res_primal <= tolerances.primal and \
					   res_dual <= tolerances.dual

		self.assertEqual( converged, converged_py )
		self.free_vars('dy', 'dmu')

	def test_default_settings(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
 			self.assert_default_settings(lib)

	def test_operator_gen_free(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for optype in self.op_keys:
				_, o = self.register_pogs_operator(lib, optype)
				self.assertEqual(type(o), lib.operator_p)
				self.free_var('o')

	def test_pogs_init_finish(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			NO_DEVICE_RESET = 0

			for optype in self.op_keys:
				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:
						_, o = self.register_pogs_operator(lib, optype, 'o')
						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.assertCall( lib.pogs_finish(
								solver, NO_DEVICE_RESET) )

						self.free_var('o')
						self.assertCall( lib.ok_device_reset() )

	def test_pogs_private_api(self):
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			elif not lib.full_api_accessible:
				continue
			self.register_exit(lib.ok_device_reset)

			NO_DEVICE_RESET = 0
			for optype in self.op_keys:
				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:

						hdl = self.register_blas_handle(lib, 'hdl')
						f, f_py, g, g_py = self.gen_registered_pogs_fns(
								lib, m, n)
						f_list = [lib.function(*f_) for f_ in f_py]
						g_list = [lib.function(*g_) for g_ in g_py]

						A, o = self.register_pogs_operator(lib, optype, 'o')

						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.register_solver('solver', solver, lib.pogs_finish)

						output, info, settings = self.gen_pogs_params(
								lib, m, n)

						local_vars = self.PogsVariablesLocal(m, n, lib.pyfloat)

						res = lib.pogs_residuals()
						tols = lib.pogs_tolerances()
						obj = lib.pogs_objectives()
						self.assertCall( lib.initialize_conditions(
								obj, res, tols, settings, m, n) )

						# test (coldstart) solver calls
						z = solver.contents.z
						W = solver.contents.W
						rho = solver.contents.rho
						self.assert_pogs_equilibration(lib, solver, A, o,
													   local_vars)
						self.assert_pogs_projector(lib, hdl, W.contents.P, o)
						self.assert_pogs_scaling(lib, solver, f, f_py, g, g_py,
												 local_vars)
						self.assert_pogs_primal_update(lib, solver, local_vars)
						self.assert_pogs_prox(lib, hdl, solver, f, f_py, g,
											  g_py, local_vars)
						self.assert_pogs_primal_project(lib, hdl, solver,
														local_vars)
						self.assert_pogs_dual_update(lib, hdl, solver,
													 local_vars)
						self.assert_pogs_check_convergence(lib, hdl, solver,
														   f_list, g_list, obj,
														   res, tols,
														   local_vars)
						self.assert_pogs_adapt_rho(lib, solver, res, tols,
												   local_vars)
						self.assert_pogs_unscaling(lib, output, solver,
												   local_vars)

						# test (warmstart) variable initialization:
						x0, x0_ptr = self.gen_py_vector(lib, n, random=True)
						nu0, nu0_ptr = self.gen_py_vector(lib, m, random=True)

						settings.x0 = x0_ptr
						settings.nu0 = nu0_ptr

						self.assertCall( lib.update_settings(
								solver.contents.settings,
								byref(settings)) )
						self.assertCall( lib.initialize_variables(solver) )
						self.assert_pogs_warmstart(lib, solver, local_vars, x0,
												   nu0)
						self.free_vars('solver', 'o', 'f', 'g', 'hdl')
						self.assertCall( lib.ok_device_reset() )

	def test_pogs_call(self):
		"""abstract operator pogs: pogs_solve() call"""
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			NO_DEVICE_RESET = 0
			for optype in self.op_keys:
				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:
						f, f_py, g, g_py = self.gen_registered_pogs_fns(
								lib, m, n)
						f_list = [lib.function(*f_) for f_ in f_py]
						g_list = [lib.function(*g_) for g_ in g_py]

						A, o = self.register_pogs_operator(lib, optype, 'o')

						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.register_solver('solver', solver, lib.pogs_finish)

						output, info, settings = self.gen_pogs_params(
								lib, m, n)
						settings.verbose = 1

						local_vars = self.PogsVariablesLocal(m, n, lib.pyfloat)

						self.assertCall( lib.pogs_solve(solver, f, g, settings,
														info, output.ptr) )
						self.free_vars('solver', 'o')

						if info.converged:
							self.assert_pogs_convergence(
									A, settings, output, gpu=gpu,
									single_precision=single_precision)

						self.free_vars('f', 'g')
						self.assertCall( lib.ok_device_reset() )

	def test_pogs_call_unified(self):
		"""abstract operator pogs: pogs() call"""
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			NO_DEVICE_RESET = 0
			for optype in self.op_keys:
				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:
						f, f_py, g, g_py = self.gen_registered_pogs_fns(
								lib, m, n)
						f_list = [lib.function(*f_) for f_ in f_py]
						g_list = [lib.function(*g_) for g_ in g_py]

						A, o = self.register_pogs_operator(lib, optype, 'o')

						output, info, settings = self.gen_pogs_params(
								lib, m, n)

						local_vars = self.PogsVariablesLocal(m, n, lib.pyfloat)

						self.assertCall( lib.pogs(o, f, g, settings, info,
												  output.ptr, DIRECT,
												  EQUILNORM, NO_DEVICE_RESET) )

						self.free_var('o')

						if info.converged:
							self.assert_pogs_convergence(
									A, settings, output, gpu=gpu,
									single_precision=single_precision)

						self.free_vars('f', 'g')
						self.assertCall( lib.ok_device_reset() )

	def test_pogs_warmstart(self):
		"""abstract operator pogs: warm start testing"""
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			NO_DEVICE_RESET = 0
			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5
			ATOLM = RTOL * m**0.5

			for optype in self.op_keys:
				for DIRECT in [0, 1]:

					EQUILNORM = 1.
					f, f_py, g, g_py = self.gen_registered_pogs_fns(
							lib, m, n)
					f_list = [lib.function(*f_) for f_ in f_py]
					g_list = [lib.function(*g_) for g_ in g_py]

					# problem matrix
					A, o = self.register_pogs_operator(lib, optype, 'o')

					# solver
					solver = lib.pogs_init(o, DIRECT, EQUILNORM)
					self.register_solver('solver', solver, lib.pogs_finish)
					output, info, settings = self.gen_pogs_params(lib, m, n)

					# warm start settings
					x0, x0_ptr = self.gen_py_vector(lib, n, random=True)
					nu0, nu0_ptr = self.gen_py_vector(lib, m, random=True)
					settings.maxiter = 0
					settings.x0 = x0_ptr
					settings.nu0 = nu0_ptr
					settings.warmstart = 1

					print "\nwarm start variable loading test (0 iters)"
					self.assertCall( lib.pogs_solve(solver, f, g, settings,
													info, output.ptr) )
					self.assertEqual( info.err, 0 )
					self.assertTrue( info.converged or
									 info.k >= settings.maxiter )

					# CHECK VARIABLE INPUT
					local_vars = self.PogsVariablesLocal(m, n, lib.pyfloat)
					self.assert_pogs_warmstart(lib, solver, local_vars, x0,
											   nu0)

					# WARMSTART SOLVE SEQUENCE
					self.assert_warmstart_sequence(lib, solver, f, g, settings,
												   info, output)

					self.free_vars('solver', 'o', 'f', 'g')
					self.assertCall( lib.ok_device_reset() )