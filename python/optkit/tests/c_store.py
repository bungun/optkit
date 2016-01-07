from optkit.api import backend
from optkit.api import Vector, Matrix, FunctionVector
from optkit.api import CPogsTypes

from optkit.types import ok_enums
from optkit.tests.defs import TEST_EPS, rand_arr
from optkit.utils.pyutils import println, pretty_print, printvoid, \
	var_assert, array_compare
from optkit.utils.proxutils import prox_eval_python, func_eval_python
from sys import argv
from numpy import ndarray, zeros, copy as np_copy, dot as np_dot
from numpy.linalg import norm
from ctypes import c_void_p
from subprocess import call
from os import path 

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


def main(m , n, A_in=None, VERBOSE_TEST=True):
	PRINT = println if VERBOSE_TEST else printvoid
	PPRINT = pretty_print if VERBOSE_TEST else printvoid

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

	# ------------------------------------------------------------ #
	# --------------------------- POGS --------------------------- #
	# ------------------------------------------------------------ #
	settings = SolverSettings()
	info = SolverInfo()
	output = SolverOutput(m, n)
	layout = ok_enums.CblasRowMajor if A.flags.c_contiguous \
		else ok_enums.CblasColMajor

	f = FunctionVector(m, b=1, h='Abs')
	g = FunctionVector(n, h='IndGe0')


	solver = lib.pogs_init(ndarray_pointer(A), m, n, layout,
		lib.enums.EquilSinkhorn)

	PPRINT("POGS SOLVE")
	lib.pogs_solve(solver, f.c, g.c, settings.c, info.c, output.c)


	k_orig = info.c.k

	mindim = min(m, n)
	A_equil = FLOAT_CAST(zeros((m, n)))
	if lib.direct:
		LLT = FLOAT_CAST(zeros((mindim, mindim)))
		LLT_ptr = ndarray_pointer(LLT)
	else:
		LLT = c_void_p()
		LLT_ptr = LLT
	order = ok_enums.CblasRowMajor if A_equil.flags.c_contiguous \
		else ok_enums.CblasRowMajor
	d = FLOAT_CAST(zeros(m))
	e = FLOAT_CAST(zeros(n))
	z = FLOAT_CAST(zeros(m + n))
	z12 = FLOAT_CAST(zeros(m + n))
	zt = FLOAT_CAST(zeros(m + n))
	zt12 = FLOAT_CAST(zeros(m + n))
	zprev = FLOAT_CAST(zeros(m + n))
	rho = zeros(1, dtype=FLOAT_CAST)


	if A_equil.flags.c_contiguous != LLT.flags.c_contiguous:
		Warning("A_equil and LLT do not have the same matrix layout.")

	PPRINT("EXTRACT SOLVER")
	lib.pogs_extract_solver(solver, ndarray_pointer(A_equil), 
		LLT_ptr, ndarray_pointer(d), ndarray_pointer(e),
		ndarray_pointer(z), ndarray_pointer(z12), ndarray_pointer(zt),
		ndarray_pointer(zt12), ndarray_pointer(zprev), 
		ndarray_pointer(rho), order)

	xrand = rand_arr(n)
	Ax = A_equil.dot(xrand)
	DAEx = d * A.dot(e * xrand)
	PRINT("A_{equil}x_{rand} - D^{-1}AE^{-1}x_{rand}:")
	PRINT(Ax - DAEx)
	assert array_compare(Ax, DAEx, eps=TEST_EPS)

	if lib.full_api_accessible:
		PRINT("extracted rho: {}\tsolver rho: {}".format(rho, solver.contents.rho))
		assert rho == solver.contents.rho


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

	assert s3.info.c.k < s2.info.c.k
	assert abs(s2.info.c.obj - s.info.c.obj) <= \
		max(10 * s.settings.c.reltol,
			10 * s.settings.c.reltol * abs(s.info.c.obj))
	assert abs(s3.info.c.obj - s.info.c.obj) <= \
		max(10 * s.settings.c.reltol,
			10 * s.settings.c.reltol * abs(s.info.c.obj))

	return True


def test_cstore(*args,**kwargs):
	print("\n\n")
	pretty_print("C POGS SAVE/LOAD TESTING ...", '#')
	print("\n\n")

	args = list(args)
	verbose = '--verbose' in args
	
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (20, 10)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	assert main(m, n, A_in=A, VERBOSE_TEST=verbose)
	if isinstance(A, ndarray): A = A.T
	assert main(n, m, A_in=A, VERBOSE_TEST=verbose)

	print("\n\n")
	pretty_print("... passed", '#')
	print("\n\n")

	return True

if __name__ == '__main__':
	args = []
	kwargs = {}

	args += argv
	if '--size' in argv:
		pos = argv.index('--size')
		if len(argv) > pos + 2:
			kwargs['shape']=(int(argv[pos+1]),int(argv[pos+2]))
	if '--file' in argv:
		pos = argv.index('--file')
		if len(argv) > pos + 1:
			kwargs['file']=str(argv[pos+1])

	test_cstore(*args, **kwargs)

