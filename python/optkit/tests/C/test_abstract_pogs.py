import os
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from ctypes import c_void_p, byref, cast, addressof
from optkit.libs import PogsAbstractLibs
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.base import OptkitCPogsTestCase, OptkitCOperatorTestCase

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
		self.assertTrue(any(libs))

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

	def assert_pogs_equilibration(self, lib, solver, A, opA, localvars):
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

		self.assertVectorEqual( A_eqx, DAEx, ATOLM, RTOL)
		self.free_vars('x', 'y')

	def assert_pogs_projector(self, lib, blas_handle, solver, opA):
		DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
		RTOL = 10**(-DIGITS)
		ATOLM = RTOL * opA.contents.size1**0.5

		m, n = (opA.contents.size1, opA.contents.size2)

		x_in, x_in_py, x_in_ptr = self.register_vector(lib, n, 'x_in')
		y_in, y_in_py, y_in_ptr = self.register_vector(lib, n, 'y_in')
		x_out, x_out_py, x_out_ptr = self.register_vector(lib, n, 'x_out')
		y_out, y_out_py, y_out_ptr = self.register_vector(lib, n, 'y_out')
		y_c, y_, y_ptr = self.register_vector(lib, n, 'y_c')


		x_in_py += np.random.rand(n).astype(lib.pyfloat)
		y_in_py += np.random.rand(m).astype(lib.pyfloat)
		self.assertCall( lib.vector_memcpy_va(x_in, x_in_ptr, 1) )
		self.assertCall( lib.vector_memcpy_va(y_in, y_in_ptr, 1) )

		# project (x_in, y_in) onto {(x, y) | y=Ax} to obtain (x_out, y_out)
		P = solver.contents.W.contents.P
		self.assertCall( P.contents.project(P.contents.data, x_in, y_in, x_out,
											y_out, 1e-3 * RTOL) )

		self.load_to_local(lib, x_out_py, x_out)
		self.load_to_local(lib, y_out_py, y_out)

		# form y_ = Ax_out
		self.assertCall( opA.contents.apply(opA.contents.data, x_out, y_c) )
		self.load_to_local(lib, y_, y_c)

		# test Ax_out ~= y_out
		self.assertVecEqual( y_, y_out_py, ATOLM, RTOL )

		self.free_vars('x_in', 'y_in', 'x_out', 'y_out', 'y_c')

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
						self.assertNotEqual(solver, 0)
						self.assertCall( lib.pogs_finish(solver,
														 NO_DEVICE_RESET) )

						self.free_var('o')
						self.assertCall( lib.ok_device_reset() )

	def test_pogs_private_api(self):
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
						if lib.full_api_accessible:
							continue

						hdl = self.register_blas_handle(lib, 'hdl')
						f, f_py, g, g_py = self.gen_registered_pogs_test_vars(
								lib, m, n)

						A, o = self.register_pogs_operator(lib, optype, 'o')

						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.register_solver('solver', solver, lib.pogs_finish)

						output, info, settings = self.gen_pogs_params(
								lib, m, n)

						localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)

						res = lib.pogs_residuals(0, 0, 0)
						tols = lib.make_tolerances(settings, m, n)
						obj = lib.pogs_objectives(0, 0, 0)
						zz = slvr.contents.z.contents

						# test (coldstart) solver calls
						z = solver.contents.z
						W = solver.contents.W
						rho = solver.contents.rho
						self.assert_pogs_equilibration(lib, solver, A, o,
													   localvars)
						self.assert_pogs_projector(lib, hdl, solver, o)
						self.assert_pogs_scaling(lib, solver, W, f, f_py, g,
												 g_py, localvars)
						self.assert_pogs_primal_update(lib, z, localvars)
						self.assert_pogs_prox(lib, hdl, z, W, rho, f, f_py, g,
											  g_py, localvars)
						self.assert_pogs_primal_project(lib, hdl, z, settings,
														localA, localvars)
						self.assert_pogs_dual_update(lib, hdl, z,
													 localvars)
						self.assert_pogs_check_convergence(lib, hdl, solver,
														   f_list, g_list, obj,
														   res, tols, localA,
														   localvars)
						self.assert_pogs_adapt_rho(lib, solver, settings, res,
												   tols, localvars)
						self.assert_pogs_unscaling(lib, solver, output,
												   localvars)

						# test (warmstart) variable initialization:
						settings.x0 = x_rand.ctypes.data_as(lib.ok_float_p)
						settings.nu0 = nu_rand.ctypes.data_as(lib.ok_float_p)
						self.assertCall( lib.update_settings(
									solver.contents.settings, settings) )
						self.assertCall( lib.initialize_variables(solver) )
						self.assert_pogs_warmstart(lib, rho, z, W, settings,
												   localA, localvars)

						self.free_vars('solver', 'o', 'f', 'g', 'hdl')
						self.assertCall( lib.ok_device_reset() )

	def test_pogs_call(self):
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
						f, f_py, g, g_py = self.gen_registered_pogs_test_vars(
								lib, m, n)
						f_list = [lib.function(*f_) for f_ in f_py]
						g_list = [lib.function(*g_) for g_ in g_py]

						A, o = self.register_pogs_operator(lib, optype, 'o')

						solver = lib.pogs_init(o, DIRECT, EQUILNORM)
						self.register_solver('solver', solver, lib.pogs_finish)

						output, info, settings = self.gen_pogs_params(
								lib, m, n)
						settings.verbose = 0

						localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)

						self.assertCall( lib.pogs_solve(solver, f, g, settings,
														info, output.ptr) )
						self.free_vars('solver', 'o')

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
			self.register_exit(lib.ok_device_reset)

			NO_DEVICE_RESET = 0
			for optype in self.op_keys:
				for DIRECT in [0, 1]:
					for EQUILNORM in [1., 2.]:
						f, f_py, g, g_py = self.gen_registered_pogs_test_vars(
								lib, m, n)
						f_list = [lib.function(*f_) for f_ in f_py]
						g_list = [lib.function(*g_) for g_ in g_py]

						A, o = self.register_pogs_operator(lib, optype, 'o')

						output, info, settings = self.gen_pogs_params(
								lib, m, n)

						localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)

						self.assertCall( lib.pogs(o, f, g, settings, info,
												  output.ptr, DIRECT,
												  EQUILNORM, NO_DEVICE_RESET) )

						self.free_var('o')

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
			self.register_exit(lib.ok_device_reset)

			NO_DEVICE_RESET = 0
			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5
			ATOLM = RTOL * m**0.5

			for optype in self.op_keys:
				for DIRECT in [0, 1]:

					EQUILNORM = 1.

					self.assertEqual(lib.blas_make_handle(byref(hdl)), 0)

					f, f_py, g, g_py = self.gen_registered_pogs_test_vars(
							lib, m, n)
					f_list = [lib.function(*f_) for f_ in f_py]
					g_list = [lib.function(*g_) for g_ in g_py]

					x_rand = self.gen_py_vector(lib, n)
					nu_rand = self.gen_py_vector(lib, m)

					# problem matrix
					A, o = self.register_pogs_operator(lib, optype, 'o')

					# solver
					solver = lib.pogs_init(o, DIRECT, EQUILNORM)
					self.register_solver('solver', solver, lib.pogs_finish)
					output, info, settings = self.gen_pogs_params(lib, m, n)

					# warm start settings
					settings.maxiter = 0
					settings.x0 = x_rand.ctypes.data_as(lib.ok_float_p)
					settings.nu0 = nu_rand.ctypes.data_as(lib.ok_float_p)
					settings.warmstart = 1

					print "\nwarm start variable loading test (0 iters)"
					self.assertCall( lib.pogs_solve(solver, f, g, settings,
													info, output.ptr) )
					self.assertEqual( info.err, 0 )
					self.assertTrue( info.converged or
									 info.k >= settings.maxiter )

					# CHECK VARIABLE INPUT
					localvars = self.PogsVariablesLocal(m, n, lib.pyfloat)
					self.assert_pogs_warmstart(
							lib, solver.contents.rho, solver.contents.z,
							solver.contents.M, settings, localA, localvars)

					# WARMSTART SOLVE SEQUENCE
					self.assert_warmstart_sequence(lib, solver, f, g, settings,
												   info, output)

					self.free_vars('solver', 'o', 'f', 'g')
					self.assertCall( lib.ok_device_reset() )