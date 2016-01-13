from traceback import format_exc
from sys import argv
from numpy import ndarray, zeros, copy as np_copy, dot as np_dot, float32
from numpy.linalg import norm
from ctypes import c_void_p
from subprocess import call
from os import path 
from optkit.types import ok_enums
from optkit.utils.pyutils import println, pretty_print, printvoid, \
	var_assert, array_compare
from optkit.utils.proxutils import prox_eval_python, func_eval_python
from optkit.tests.defs import gen_test_defs

def main(errors, m , n, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import Vector, Matrix
		from optkit.api import CPogsTypes
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

		ndarray_pointer = backend.lowtypes.ndarray_pointer
		lib = backend.pogs
		FLOAT_CAST = backend.lowtypes.FLOAT_CAST

		AdaptiveRhoParameters = lib.adapt_params
		PogsObjectives = lib.pogs_objectives
		PogsResiduals = lib.pogs_residuals
		PogsTolerances = lib.pogs_tolerances

		SolverSettings = CPogsTypes.SolverSettings
		SolverInfo = CPogsTypes.SolverInfo
		SolverOutput = CPogsTypes.SolverOutput
		Solver = CPogsTypes.Solver
		Objective = CPogsTypes.Objective

		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid



		# ------------------------------------------------------------ #
		# ------------------------ test setup ------------------------ #
		# ------------------------------------------------------------ #
		PRINT("\n")

		if isinstance(A_in, ndarray):
			A = A_in.astype(FLOAT_CAST)
			(m, n) = A.shape
			PRINT("(using provided matrix)")
		else:
			A = RAND_ARR(m,n)
			PRINT("(using random matrix)")

		pretty_print("{} MATRIX".format("SKINNY" if m >= n else "FAT"), '=')
		print "(m = {}, n = {})".format(m, n)



		# ------------------------------------------------------------ #
		# --------------------------- POGS --------------------------- #
		# ------------------------------------------------------------ #
		settings = SolverSettings()
		info = SolverInfo()
		output = SolverOutput(m, n)
		layout = ok_enums.CblasRowMajor if A.flags.c_contiguous \
			else ok_enums.CblasColMajor

		f = Objective(m, b=1, h='Abs')
		g = Objective(n, h='IndGe0')


		solver = lib.pogs_init(ndarray_pointer(A), m, n, layout,
			lib.enums.EquilSinkhorn)

		PPRINT("POGS SOLVE")
		lib.pogs_solve(solver, f.c, g.c, settings.c, info.c, output.c)


		k_orig = info.c.k

		mindim = min(m, n)
		A_equil = zeros((m, n)).astype(FLOAT_CAST)
		if lib.direct:
			LLT = zeros((mindim, mindim)).astype(FLOAT_CAST)
			LLT_ptr = ndarray_pointer(LLT)
		else:
			LLT = c_void_p()
			LLT_ptr = LLT
		order = ok_enums.CblasRowMajor if A_equil.flags.c_contiguous \
			else ok_enums.CblasRowMajor
		d = zeros(m).astype(FLOAT_CAST)
		e = zeros(n).astype(FLOAT_CAST)
		z = zeros(m + n).astype(FLOAT_CAST)
		z12 = zeros(m + n).astype(FLOAT_CAST)
		zt = zeros(m + n).astype(FLOAT_CAST)
		zt12 = zeros(m + n).astype(FLOAT_CAST)
		zprev = zeros(m + n).astype(FLOAT_CAST)
		rho = zeros(1).astype(FLOAT_CAST)


		if A_equil.flags.c_contiguous != LLT.flags.c_contiguous:
			Warning("A_equil and LLT do not have the same matrix layout.")

		PPRINT("EXTRACT SOLVER")
		lib.pogs_extract_solver(solver, ndarray_pointer(A_equil), 
			LLT_ptr, ndarray_pointer(d), ndarray_pointer(e),
			ndarray_pointer(z), ndarray_pointer(z12), ndarray_pointer(zt),
			ndarray_pointer(zt12), ndarray_pointer(zprev), 
			ndarray_pointer(rho), order)

		xrand = RAND_ARR(n)
		Ax = A_equil.dot(xrand)
		DAEx = d * A.dot(e * xrand)
		PRINT("A_{equil}x_{rand} - D^{-1}AE^{-1}x_{rand}:")
		PRINT(Ax - DAEx)
		assert array_compare(Ax, DAEx, eps=TEST_EPS)

		if lib.full_api_accessible:
			PRINT("extracted rho: {}\tsolver rho: {}".format(rho, solver.contents.rho))
			assert rho[0] == solver.contents.rho


		PRINT("(DESTROY SOLVER)")
		lib.pogs_finish(solver)

		PPRINT("LOAD EXTRACTED SOLVER") 

		solver = lib.pogs_load_solver(ndarray_pointer(A_equil), 
			LLT_ptr, ndarray_pointer(d), ndarray_pointer(e),
			ndarray_pointer(z), ndarray_pointer(z12), ndarray_pointer(zt),
			ndarray_pointer(zt12), ndarray_pointer(zprev), rho[0],
			m, n, order)

		PRINT("SOLVE")
		settings.c.resume = 1
		lib.pogs_solve(solver, f.c, g.c, settings.c, info.c, output.c)
		PRINT("SOLVER ITERATIONS:", info.c.k)
		assert info.c.k <= k_orig
		lib.pogs_finish(solver)

		PPRINT("TEST PYTHON BINDINGS:")
		PRINT("MAKE SOLVER")
		s = Solver(A)
		PRINT("SAVE SOLVER BEFORE RUN")
		s.save(path.abspath('.'),'c_solve_test')
		PRINT("RUN SOLVER")
		s.solve(f, g, resume=0)

		PRINT("LOAD SAVED SOLVER")
		s2 = Solver(A, 'no_init')
		s2.load(path.abspath('.'), 'c_solve_test')
		PRINT("RUN LOADED SOLVER")
		s2.solve(f, g, resume=0)
		PRINT("SAVE SOLVER AFTER RUN")
		call(['rm', 'c_solve_test.npz'])
		s2.save(path.abspath('.'),'c_solve_test')

		s3 = Solver(A, 'no_init')
		s3.load(path.abspath('.'), 'c_solve_test')
		PRINT("RUN 2nd LOADED SOLVER (warmstart)")
		s3.solve(f, g, resume=1)
		call(['rm', 'c_solve_test.npz'])

		PRINT("ITERATIONS:")
		PRINT("first solver: {}".format(s.info.c.k))
		PRINT("second solver: {}".format(s2.info.c.k))
		PRINT("third solver: {}".format(s3.info.c.k))

		PRINT("OBJECTIVE VALUES:")
		PRINT("first solver: {}".format(s.info.c.obj))
		PRINT("second solver: {}".format(s2.info.c.obj))
		PRINT("third solver: {}".format(s3.info.c.obj))

		FAC = 30 if backend.lowtypes.FLOAT_CAST == float32 else 10
		assert s3.info.c.k <= s2.info.c.k
		assert abs(s2.info.c.obj - s.info.c.obj) <= \
			max(FAC * s.settings.c.reltol,
				FAC * s.settings.c.reltol * abs(s.info.c.obj))
		assert abs(s3.info.c.obj - s.info.c.obj) <= \
			max(FAC * s.settings.c.reltol,
				FAC * s.settings.c.reltol * abs(s.info.c.obj))

		return True

	except:
		errors.append(format_exc())
		return False

def test_cstore(errors, *args,**kwargs):
	print("\n\n")
	pretty_print("C POGS SAVE/LOAD TESTING ...", '#')
	print("\n\n")

	args = list(args)
	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64
	
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (20, 10)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	success = main(errors, m, n, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)
	if isinstance(A, ndarray): A = A.T
	success &= main(errors, n, m, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)

	if success:
		print("\n\n")
		pretty_print("... passed", '#')
		print("\n\n")
	else:
		print("\n\n")
		pretty_print("... failed", '#')
		print("\n\n")
	return success