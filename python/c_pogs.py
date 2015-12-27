from ctypes import CDLL, POINTER, Structure, c_void_p, c_uint, \
					c_int, c_size_t, byref
from os import uname
from optkit.api import backend


from optkit.api import Vector, Matrix, FunctionVector
from optkit.types import ok_enums
from optkit.tests.defs import HLINE, TEST_EPS, rand_arr
from optkit.utils.pyutils import println, pretty_print, printvoid, \
	var_assert, array_compare
from optkit.utils.proxutils import prox_eval_python, func_eval_python
from sys import argv
from numpy import ndarray, zeros, copy as np_copy, dot as np_dot
from numpy.linalg import norm
from os import path

ok_float = backend.lowtypes.ok_float
ok_float_p = backend.lowtypes.ok_float_p
vector_p = backend.lowtypes.vector_p
matrix_p = backend.lowtypes.matrix_p
function_vector_p = backend.lowtypes.function_vector_p
ndarray_pointer = backend.lowtypes.ndarray_pointer
hdl = backend.dense_blas_handle

DEVICE = backend.device
PRECISION = backend.precision
LAYOUT = backend.layout + '_' if backend.layout != '' else ''
EXT = 'dylib' if uname()[0] == 'Darwin' else 'so'

def main(m = 10, n = 5, A_in=None, VERBOSE_TEST=True):
	PRINT = println if VERBOSE_TEST else printvoid
	PPRINT = pretty_print if VERBOSE_TEST else printvoid
	HOME = path.dirname(path.abspath(__file__))

	# ------------------------------------------------------------ #
	# ---------------------- libpogs prep ------------------------ #
	# ------------------------------------------------------------ #

	libpath = path.join(path.join(HOME, '..', 'build'),
	 'libpogs_dense_{}{}{}.{}'.format(LAYOUT, DEVICE, PRECISION, EXT))
	lib = CDLL(path.abspath(libpath))


	class PogsSettings(Structure):
		_fields_ = [('alpha', ok_float),
					('rho', ok_float),
					('abstol', ok_float),
					('reltol', ok_float),
					('maxiter', c_uint),
					('verbose', c_uint),
					('adaptiverho', c_int),
					('gapstop', c_int),
					('warmstart', c_int),
					('resume', c_int),
					('x0', ok_float_p),
					('nu0', ok_float_p)]

	pogs_settings_p = POINTER(PogsSettings)

	class PogsInfo(Structure):
		_fields_ = [('err', c_int),
					('converged', c_int),
					('k', c_uint),
					('obj', ok_float),
					('rho', ok_float),
					('setup_time', ok_float),
					('solve_time', ok_float)]

	pogs_info_p = POINTER(PogsInfo)

	class PogsOutput(Structure):
		_fields_ = [('x', ok_float_p),
					('y', ok_float_p),
					('mu', ok_float_p),
					('nu', ok_float_p)]		

	pogs_output_p = POINTER(PogsOutput)


	lib.set_default_settings.argtypes = [pogs_settings_p]
	lib.pogs_init.argtypes = [ok_float_p, 
		c_size_t, c_size_t, c_uint, c_uint]
	lib.pogs_solve.argtypes = [c_void_p, function_vector_p, function_vector_p, 
		pogs_settings_p, pogs_info_p, pogs_output_p]
	lib.pogs_finish.argtypes = [c_void_p]
	lib.pogs.argtypes = [ok_float_p, function_vector_p, function_vector_p,
		pogs_settings_p, pogs_info_p, pogs_output_p, c_uint, c_uint]
	lib.pogs_load_solver.argtypes = [ok_float_p, 
		ok_float_p, ok_float_p, ok_float_p, ok_float_p, ok_float_p, 
		ok_float_p, ok_float_p, ok_float_p, ok_float, 
		c_size_t, c_size_t, c_uint]
	lib.pogs_extract_solver.argtypes = [c_void_p, ok_float_p, 
		ok_float_p, ok_float_p, ok_float_p, ok_float_p, ok_float_p, 
		ok_float_p, ok_float_p, ok_float_p, ok_float_p, c_uint]

	lib.set_default_settings.restype = None
	lib.pogs_init.restype = c_void_p
	lib.pogs_solve.restype = None
	lib.pogs_finish.restype = None
	lib.pogs.restype = None
	lib.pogs_load_solver.restype = c_void_p
	lib.pogs_extract_solver.restype = None


	lib.private_api_accessible.restype = c_int
	full_api_accessible = lib.private_api_accessible()


	if not full_api_accessible:
		AdaptiveRhoParameters = None
		adapt_params_p = None
		PogsBlockVector = None
		block_vector_p = None
		PogsResiduals = None
		pogs_residuals_p = None
		PogsTolerances = None
		pogs_tolerances_p = None
		PogsObjectives = None
		pogs_objectives_p = None
		PogsMatrix = None
		pogs_matrix_p = None
		PogsVariables = None
		pogs_variables_p = None
		PogsSolver = None
		pogs_solver_p = None
		print str('libpogs compiled without flag OPTKIT_DEBUG_PYTHON, '
			'python cannot access libpogs private methods.\n'
			'Testing public API calls only.')
	else:

		class AdaptiveRhoParameters(Structure):
			_fields_ = [('delta', ok_float),
						('l', ok_float),
						('u', ok_float),
						('xi', ok_float)]

		adapt_params_p = POINTER(AdaptiveRhoParameters)

		class PogsBlockVector(Structure):
			_fields_ = [('size', c_size_t),
						('m', c_size_t),
						('n', c_size_t),
						('x', vector_p),
						('y', vector_p),
						('vec', vector_p)]

		block_vector_p = POINTER(PogsBlockVector)


		class PogsResiduals(Structure):
			_fields_ = [('primal', ok_float),
						('dual', ok_float),
						('gap', ok_float)]

		pogs_residuals_p = POINTER(PogsResiduals)


		class PogsTolerances(Structure):
			_fields_ = [('primal', ok_float),
						('dual', ok_float),
						('gap', ok_float),
						('reltol', ok_float),
						('abstol', ok_float),
						('atolm', ok_float),
						('atoln', ok_float),
						('atolmn', ok_float)]

		pogs_tolerances_p = POINTER(PogsTolerances)

		class PogsObjectives(Structure):
			_fields_ = [('primal', ok_float),
						('dual', ok_float),
						('gap', ok_float)]

		pogs_objectives_p = POINTER(PogsObjectives)

		class PogsMatrix(Structure):
			_fields_ = [('A', matrix_p),
						('P', c_void_p),
						('d', vector_p),
						('e', vector_p),
						('normA', ok_float),
						('skinny', c_int),
						('normalized', c_int),
						('equilibrated', c_int)]

		pogs_matrix_p = POINTER(PogsMatrix)

		class PogsVariables(Structure):
			_fields_ = [('primal', block_vector_p),
						('primal12', block_vector_p),
						('dual', block_vector_p),
						('dual12', block_vector_p),
						('prev', block_vector_p),
						('temp', block_vector_p),
						('m', c_size_t),
						('n', c_size_t)]

		pogs_variables_p = POINTER(PogsVariables)



		class PogsSolver(Structure):
			_fields_ = [('M', pogs_matrix_p),
						('z', pogs_variables_p),
						('f', function_vector_p),
						('g', function_vector_p),
						('rho', ok_float),
						('settings', pogs_settings_p),
						('linalg_handle', c_void_p),
						('init_time', ok_float)]

		pogs_solver_p = POINTER(PogsSolver)

		lib.scale_problem.argtypes = [function_vector_p, function_vector_p,
			vector_p, vector_p]
		lib.initialize_variables.argtypes = [pogs_solver_p]
		lib.pogs_solver_loop.argtypes = [pogs_solver_p, pogs_info_p]
		lib.make_tolerances.argtypes = [pogs_settings_p, 
			c_size_t, c_size_t]
		lib.set_prev.argtypes = [pogs_variables_p]
		lib.prox.argtypes = [c_void_p, function_vector_p, 
			function_vector_p, pogs_variables_p, ok_float]
		lib.project_primal.argtypes = [c_void_p, c_void_p, 
			pogs_variables_p, ok_float]
		lib.update_dual.argtypes = [c_void_p, pogs_variables_p, ok_float]
		lib.check_convergence.argtypes = [c_void_p, pogs_solver_p,
			pogs_objectives_p, pogs_residuals_p, pogs_tolerances_p]
		lib.adaptrho.argtypes = [pogs_solver_p, adapt_params_p, 
			pogs_residuals_p, pogs_tolerances_p, c_uint]
		lib.copy_output.argtypes = [pogs_solver_p, pogs_output_p]

		lib.scale_problem.restype = None
		lib.initialize_variables.restype = None
		lib.pogs_solver_loop.restype = None
		lib.make_tolerances.restype = PogsTolerances
		lib.set_prev.restype = None 
		lib.prox.restype = None
		lib.project_primal.restype = None 
		lib.update_dual.restype = None
		lib.check_convergence.restype = c_int
		lib.adaptrho.restype = None
		lib.copy_output.restype = None

		lib.is_direct.restype = c_int
		lib.pogs_init.restype = pogs_solver_p
		lib.pogs_load_solver.restype = pogs_solver_p

	# ------------------------------------------------------------ #
	# ------------------------ test setup ------------------------ #
	# ------------------------------------------------------------ #
	PRINT("\n")

	if isinstance(A_in, ndarray):
		A = A_in
		(m, n) = A.shape
		PRINT("(using provided matrix)")
	else:
		A = rand_arr(m,n)
		PRINT("(using random matrix)")

	pretty_print("{} MATRIX".format("SKINNY" if m >= n else "FAT"), '=')
	print "(m = {}, n = {})".format(m, n)


	class SolverSettings(object):
		def __init__(self, pogs_lib, **options):
			self.c = PogsSettings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None)
			pogs_lib.set_default_settings(self.c)
			self.update(**options)

		def update(self, **options):
			if 'alpha' in options: self.c.alpha = options['alpha']
			if 'rho' in options: self.c.rho = options['rho']
			if 'abstol' in options: self.c.abstol = options['abstol']
			if 'reltol' in options: self.c.reltol = options['reltol']
			if 'maxiter' in options: self.c.maxiter = options['maxiter']
			if 'verbose' in options: self.c.verbose = options['verbose']
			if 'adaptiverho' in options: self.c.adaptiverho = options['adaptiverho']
			if 'gapstop' in options: self.c.gapstop = options['gapstop']
			if 'resume' in options: self.c.resume = options['resume']
			if 'x0' in options: self.c.x0 = ndarray_pointer(options['x0'])
			if 'nu0' in options: self.c.nu0 = ndarray_pointer(options['nu0'])

		def __str__(self):
			return str(
				"alpha: {}\n".format(self.c.alpha) + 
				"rho: {}\n".format(self.c.rho) +
				"abstol: {}\n".format(self.c.abstol) +
				"reltol: {}\n".format(self.c.reltol) +
				"maxiter: {}\n".format(self.c.maxiter) +
				"verbose: {}\n".format(self.c.verbose) +
				"adaptiverho: {}\n".format(self.c.adaptiverho) +
				"gapstop: {}\n".format(self.c.gapstop) +
				"warmstart: {}\n".format(self.c.warmstart) +
				"resume: {}\n".format(self.c.resume) +
				"x0: {}\n".format(self.c.x0) +
				"nu0: {}\n".format(self.c.nu0))
	
	class SolverInfo(object):
		def __init__(self):
			self.c = PogsInfo(0, 0, 0, 0, 0, 0, 0)

		def __str__(self):
			return str(
				"error: {}\n".format(self.c.err) + 
				"converged: {}\n".format(self.c.converged) +
				"iterations: {}\n".format(self.c.k) +
				"objective: {}\n".format(self.c.obj) +
				"rho: {}\n".format(self.c.rho) +
				"setup time: {}\n".format(self.c.setup_time) +
				"solve time: {}\n".format(self.c.solve_time))


	class SolverOutput(object):
		def __init__(self, m, n):
			self.x = zeros(n)
			self.y = zeros(m)
			self.mu = zeros(n)
			self.nu = zeros(m)
			self.c = PogsOutput(ndarray_pointer(self.x), ndarray_pointer(self.y),
				ndarray_pointer(self.mu), ndarray_pointer(self.nu))

		def __str__(self):
			return str("x:\n{}\ny:\n{}\nmu:\n{}\nnu:\n{}\n".format(
				str(self.x), str(self.y), 
				str(self.mu), str(self.nu)))


	class OKEquilibrationEnums(object):
		EquilSinkhorn = c_uint(0).value
		EquilDenseL2 = c_uint(1).value

	equil_enums = OKEquilibrationEnums()

	# ------------------------------------------------------------ #
	# --------------------------- POGS --------------------------- #
	# ------------------------------------------------------------ #
	settings = SolverSettings(lib)
	info = SolverInfo()
	output = SolverOutput(m, n)
	layout = ok_enums.CblasRowMajor if A.flags.c_contiguous \
		else ok_enums.CblasColMajor

	f = FunctionVector(m, b=1, h='Abs')
	g = FunctionVector(n, h='IndGe0')


	solver = lib.pogs_init(ndarray_pointer(A), m, n, layout,
		equil_enums.EquilSinkhorn)

	# -------------------- private api tests---------------------- #
	if full_api_accessible:
		PPRINT('POGS PRIVATE API TESTS', '+')

		PPRINT('VERIFY INITIALIZATION:')
		A_local = zeros((m, n))
		ordA = ok_enums.CblasRowMajor if A_local.flags.c_contiguous \
			else CblasColMajor
		d_local = zeros(m)
		e_local = zeros(n)

		backend.dense.matrix_memcpy_am(ndarray_pointer(A_local),
			solver.contents.M.contents.A, ordA)
		backend.dense.vector_memcpy_av(ndarray_pointer(d_local),
			solver.contents.M.contents.d, 1)
		backend.dense.vector_memcpy_av(ndarray_pointer(e_local),
			solver.contents.M.contents.e, 1)

		PRINT('VERIFY EQULIBRATION')
		xrand = rand_arr(n)
		Ax = A_local.dot(xrand)
		DAEx = d_local * A.dot(e_local * xrand)
		PRINT("A_{equil}x - (D^{-1} A E^{-1})x")
		PRINT(Ax-DAEx)
		assert array_compare(Ax, DAEx, eps=TEST_EPS)

		PRINT('VERIFY PROJECTOR')
		x_in = Vector(rand_arr(n))
		y_in = Vector(rand_arr(m))
		x_out = Vector(zeros(n))
		y_out = Vector(zeros(m))
		x_local = zeros(n)
		y_local = zeros(m)


		if lib.is_direct():
			lib.direct_projector_project.argtypes = [c_void_p,
				c_void_p, vector_p, vector_p, vector_p, vector_p]
			lib.direct_projector_project.restype = None
			lib.direct_projector_project(backend.dense_blas_handle,
				solver.contents.M.contents.P, x_in.c, y_in.c,
				x_out.c, y_out.c)
		else:
			lib.indirect_projector_project.argtypes = [c_void_p,
				c_void_p, vector_p, vector_p, vector_p, vector_p]
			lib.indirect_projector_project.restype = None
			lib.indirect_projector_project(backend.dense_blas_handle,
				solver.contents.M.contents.P, x_in.c, y_in.c,
				x_out.c, y_out.c)


		backend.dense.vector_memcpy_av(ndarray_pointer(x_local),
			x_out.c, 1)
		backend.dense.vector_memcpy_av(ndarray_pointer(y_local),
			y_out.c, 1)


		PRINT("BEFORE:")
		PRINT("||x_in|| {}\t||y_in|| {}".format(norm(x_in.py), norm(y_in.py)))
		PRINT("||Ax_in-y_in|| {}".format(norm(A_local.dot(x_in.py) - y_in.py)))
		PRINT("AFTER:")
		PRINT("||x_out|| {}\t||y_out|| {}".format(norm(x_local), norm(y_local)))
		PRINT("||Ax_out-y_out|| {}".format(norm(A_local.dot(x_local) - y_local)))
		assert norm(A_local.dot(x_local) - y_local) < TEST_EPS

		# transfer function vectors to solver copies 
		PPRINT('TRANSFER FUNCTION VECTORS TO LOCAL COPIES')
		backend.prox.function_vector_memcpy_va(solver.contents.f, 
			f.c.objectives);
		backend.prox.function_vector_memcpy_va(solver.contents.g, 
			g.c.objectives);

		if VERBOSE_TEST:
			PRINT("f before scaling")
			backend.prox.function_vector_print(f.c)
			PRINT("g before scaling")
			backend.prox.function_vector_print(g.c)

		PPRINT('SCALE PROBLEM:')
		lib.scale_problem(f.c, g.c, 
			solver.contents.M.contents.d, solver.contents.M.contents.e)

		f.pull()
		g.pull()



		if VERBOSE_TEST:


			PRINT("\nf after scaling:")
			backend.prox.function_vector_print(f.c)
			PRINT("\nSCALING d:")
			backend.dense.vector_print(solver.contents.M.contents.d)

			PRINT("\ng after scaling:")
			backend.prox.function_vector_print(g.c)
			PRINT("\nSCALING e:")
			backend.dense.vector_print(solver.contents.M.contents.e)
			
		
		# python version of solver variables
		z_local = {'primal': zeros(m + n), 
					'primal12': zeros(m + n),
					'dual': zeros(m + n),
					'dual12': zeros(m + n),
					'prev': zeros(m + n),
					'temp': zeros(m + n),
					}


		def load_2_local(py_var, c_var):
			backend.dense.vector_memcpy_av(ndarray_pointer(py_var), c_var, 1)

		def load_all_local(z_py, z_c):
			load_2_local(z_py['primal'], z_c.contents.primal.contents.vec)
			load_2_local(z_py['primal12'], z_c.contents.primal12.contents.vec)
			load_2_local(z_py['dual'], z_c.contents.dual.contents.vec)
			load_2_local(z_py['dual12'], z_c.contents.dual12.contents.vec)
			load_2_local(z_py['prev'], z_c.contents.prev.contents.vec)
			load_2_local(z_py['temp'], z_c.contents.temp.contents.vec)

		PPRINT('OBTAIN WARMSTART VARIABLES')
		
		xrand = rand_arr(n)
		nurand = rand_arr(m)

		settings.update(x0=xrand, nu0=nurand)
		solver.contents.settings.contents.x0 = settings.c.x0
		solver.contents.settings.contents.nu0 = settings.c.nu0
		lib.initialize_variables(solver)
		load_all_local(z_local, solver.contents.z)


		PRINT("x_{input} / e - x^{0}:")
		PRINT(xrand / e_local - z_local['primal'][m:])
		PRINT("(-1/rho) * nu_{input} / d - nu^{0}:")
		PRINT(-1./solver.contents.rho * nurand / d_local - z_local['dual'][:m])

		res_p = norm(A_local.dot(z_local['primal'][m:])-z_local['primal'][:m])
		res_d = norm(A_local.T.dot(z_local['dual'][:m])+z_local['dual'][m:])
		PRINT("primal feasibility at step k=0: ", res_p)
		PRINT("dual feasibility at step k=0: ", res_p)
		assert array_compare(xrand / e_local, z_local['primal'][m:], eps=TEST_EPS)
		assert array_compare(-1./solver.contents.rho * nurand / d_local, 
			z_local['dual'][:m], eps=TEST_EPS)
		assert res_p <= TEST_EPS
		assert res_d <= TEST_EPS


		PPRINT("VARIABLE UPDATE")
		PPRINT('Z_prev := Z', '.')
		lib.set_prev(solver.contents.z)
		load_all_local(z_local, solver.contents.z)
		PRINT('z - z_prev')
		PRINT(z_local['primal'] - z_local['prev'])


		PPRINT("PROX")
		PPRINT('Z^{1/2} = Prox_{rho, f,g}(Z)', '.')
		lib.prox(backend.dense_blas_handle, f.c, g.c, solver.contents.z,
			solver.contents.rho)
		load_all_local(z_local, solver.contents.z)



		xarg_ = z_local['primal'][m:]-z_local['dual'][m:]
		yarg_ = z_local['primal'][:m]-z_local['dual'][:m]
		xout_ = prox_eval_python(g.tolist(),solver.contents.rho,xarg_)
		yout_ = prox_eval_python(f.tolist(),solver.contents.rho,yarg_)

		PRINT("\ny^{1/2} c - y^{1/2} python")
		PRINT(z_local['primal12'][:m] - yout_)
		PRINT("\nx^{1/2} c - x^{1/2} python")		
		PRINT(z_local['primal12'][m:] - xout_)
		assert array_compare(z_local['primal12'][:m], yout_, eps=TEST_EPS)
		assert array_compare(z_local['primal12'][m:], xout_, eps=TEST_EPS)


		PPRINT("PROJECT")
		lib.project_primal(backend.dense_blas_handle,
			solver.contents.M.contents.P, solver.contents.z,
			settings.c.alpha)

		load_all_local(z_local, solver.contents.z)
		PRINT("Ax^{k+1}-y^{k+1}")
		PRINT(A_local.dot(z_local['primal'][m:]) - z_local['primal'][:m])
		res = norm(A_local.dot(z_local['primal'][m:]) - z_local['primal'][:m])
		PRINT("||Ax^{k+1}-y^{k+1)||:")
		PRINT(res)
		assert res < TEST_EPS

		PPRINT("UPDATE DUAL")

		zt12_ = z_local['primal12'] - z_local['prev'] + z_local['dual']
		zt1_ = z_local['dual'] + (settings.c.alpha * z_local['primal12'] +
			(1 - settings.c.alpha) * z_local['prev']) - z_local['primal']

		lib.update_dual(backend.dense_blas_handle, solver.contents.z,
			settings.c.alpha)
		load_all_local(z_local, solver.contents.z)
		
		PPRINT("zt^{k+1/2} = z^{k+1/2} - z^k + zt^k",'.')
		PRINT("zt^{k+1/2} (C) - zt^{k+1/2} (python)")
		PRINT(z_local['dual12'] - zt12_)

		PPRINT("zt^{k} = z^{k+1/2} - z^{k+1} + zt^k",'.')
		PRINT("z^{k+1/2} (C) - z^{k+1/2} (python)")
		PRINT(z_local['dual'] - zt1_)

		assert array_compare(z_local['dual12'], zt12_, eps=TEST_EPS)
		assert array_compare(z_local['dual'], zt1_, eps=TEST_EPS)

		PPRINT("CHECK CONVERGENCE")

		res = PogsResiduals(0, 0, 0)
		eps = lib.make_tolerances(settings.c, m, n)
		obj = PogsObjectives(0, 0, 0)

		converged = lib.check_convergence(backend.dense_blas_handle,
			solver, obj, res, eps, settings.c.gapstop, 0)	
	

		obj_py = func_eval_python(g.tolist(), z_local['primal12'][m:])
		obj_py += func_eval_python(f.tolist(), z_local['primal12'][:m])
		obj_gap_py = abs(np_dot(z_local['primal12'], z_local['dual12']))
		obj_dua_py = obj_py - obj_gap_py

		PRINT("C -- primal: {}, dual {}, gap {}".format(obj.primal,
			obj.dual, obj.gap))
		PRINT("PY -- primal: {}, dual {}, gap {}".format(obj_py,
			obj_dua_py, obj_gap_py))

		assert abs(obj.gap - obj_gap_py) <= TEST_EPS

		assert abs(eps.primal - (eps.atolm + eps.reltol * norm(z_local['primal12'][:m]))) <= TEST_EPS
		assert abs(eps.dual - (eps.atoln + eps.reltol * norm(z_local['dual12'][m:]))) <= TEST_EPS

		res_p = norm(A_local.dot(z_local['primal12'][m:]) - z_local['primal12'][:m])
		res_d = norm(A_local.T.dot(z_local['dual12'][:m]) + z_local['dual12'][m:])

		assert abs(res.primal - res_p) <= TEST_EPS
		assert abs(res.dual - res_d) <= TEST_EPS
		assert abs(res.gap - abs(obj_gap_py)) <= TEST_EPS
		
		cvg_py = norm(A_local.dot(z_local['primal12'][m:]) - \
						z_local['primal12'][:m]) <= eps.primal and \
					norm(A_local.dot(z_local['dual12'][:m]) + \
								z_local['dual12'][m:]) <= eps.dual
		assert cvg_py == converged
		PRINT("Converged? ", converged )
	

		PPRINT("ADAPT RHO")
		PRINT("Adaptive rho requested?", settings.c.adaptiverho)

		rho_params = AdaptiveRhoParameters(1.05, 0, 0, 1)
		zt_before = z_local['dual']
		rho_before = solver.contents.rho
		if settings.c.adaptiverho:
			lib.adaptrho(solver, rho_params, res, eps, 1)
			load_all_local(z_local, solver.contents.z)
			zt_after = z_local['dual']
			rho_after = solver.contents.rho
			PRINT("rho_after * zt_after - rho_before * zt_before")
			PRINT(rho_after * zt_after - rho_before * zt_before)
			assert array_compare(rho_after * zt_after, 
				rho_before * zt_before, eps = TEST_EPS)



		PPRINT("VARIABLE UNSCALING")
		lib.copy_output(solver, output.c)		
		rho = solver.contents.rho

		PRINT("x^{k+1/2} * e - x_out")
		PRINT(z_local['primal12'][m:] * e_local - output.x)
		PRINT("y^{k+1/2} / d - y_out")
		PRINT(z_local['primal12'][:m] / d_local - output.y)

		PRINT("-rho * xt^{k+1/2} / e - mu_out")
		PRINT(-rho * z_local['dual12'][m:] / e_local - output.mu)
		PRINT("-rho * yt^{k+1/2} * d - nu_out")
		PRINT(-rho * z_local['dual12'][:m] * d_local - output.nu)


		assert array_compare(z_local['primal12'][m:] * e_local, output.x,
			eps=TEST_EPS)
		assert array_compare(z_local['primal12'][:m] / d_local, output.y,
			eps=TEST_EPS)
		assert array_compare(-rho * z_local['dual12'][m:] / e_local,
			output.mu, eps=TEST_EPS)
		assert array_compare(-rho * z_local['dual12'][:m] * d_local, 
			output.nu, eps=TEST_EPS)

	# ------------------- public api tests ----------------------- #
	PPRINT("POGS PUBLIC API SOLVE",'+')
	lib.pogs_solve(solver, f.c, g.c, settings.c, info.c, output.c)
	lib.pogs_finish(solver)

	print info
	# print output

	assert info.c.converged or info.c.k==settings.c.maxiter

	res_p = norm(A.dot(output.x)-output.y)
	res_d = norm(A.T.dot(output.nu)+output.mu)
	PRINT("PRIMAL FEASIBILITY: {}", res_p)
	PRINT("DUAL FEASIBILITY: {}", res_d)

	if info.c.converged:
		assert (res_p < 10 * settings.c.reltol * norm(output.y)) or \
				norm(output.y) <= 10 * settings.c.reltol

		assert (res_d <= 10 * settings.c.reltol * norm(output.mu)) or \
				norm(output.mu) <= 10 * settings.c.reltol

	PPRINT("COMPLETE",'/')




if __name__ == '__main__':
	m, n = (10, 5)
	if '--size' in argv:
		pos = argv.index('--size')
		if len(argv) > pos + 2:
			(m, n) = (int(argv[pos+1]),int(argv[pos+2]))
	verbose = '--verbose' in argv
	main(m, n, A_in=None, VERBOSE_TEST=verbose)
	main(n, m, A_in=None, VERBOSE_TEST=verbose)


