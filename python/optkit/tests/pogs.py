import numpy as np
from traceback import format_exc
from numpy.linalg import norm
from optkit.types import ok_function_enums
from optkit.tests.defs import gen_test_defs
from optkit.utils.pyutils import println, pretty_print, var_assert

def warmstart_test(errors, m=300, n=200, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)

		from optkit.api import Matrix, FunctionVector, pogs
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

		SolverState = pogs.types.SolverState
		SolverInfo = pogs.types.SolverInfo
		OutputVariables = pogs.types.OutputVariables

		if m is None: m=300
		if n is None: n=200
		if isinstance(A_in,np.ndarray):
			if len(A_in.shape)==2:
				(m,n)=A_in.shape
			else:
				A_in=None

		print '(m = {}, n = {})'.format(m, n)

		PPRINT = pretty_print #if VERBOSE_TEST else printvoid

		verbose=2 if VERBOSE_TEST else 0

		maxiter=2000
		reltol=1e-3
		A = RAND_ARR(m,n) if A_in is None else A_in.astype(backend.lowtypes.FLOAT_CAST)
		f = FunctionVector(m, h='Square', b=1)
		g = FunctionVector(n, h='IndGe0')

		PPRINT("COLD START")
		info, output, solver_state = pogs(A, f, g, 
			verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with x0")
		info, _ , _  = pogs(A, f, g, x0=output.x, 
			warmstart = True, verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with x0, rho")
		info, _ , _  = pogs(A, f, g, x0=output.x, rho=info.rho, 
			warmstart = True, verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with nu0")
		info, _ , _  = pogs(A, f, g, nu0=output.nu, 
			warmstart = True, verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with nu0, rho")
		info, _ , _  = pogs(A, f, g, nu0=output.nu, rho=info.rho, 
			warmstart = True, verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with x0 and nu0")
		info, _ , _  = pogs(A, f, g, x0=output.x, nu0=output.nu, 
			warmstart = True, verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with x0 and nu0, rho")
		info, _ , _  = pogs(A, f, g, x0=output.x, nu0=output.nu, rho=info.rho, 
			warmstart = True, verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with solver state")
		info, _ , _  = pogs(A, f, g, solver_state=solver_state,
			verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with solver state, rho")
		info, _ , _  = pogs(A, f, g, solver_state=solver_state, rho=info.rho,
			verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with solver state and 1.02x perturbed f(y)")

		f.pull()
		fobj = f.tolist()
		for i in xrange(0, m/2): fobj[i].c *= 1.02
		for i in xrange(m/2, m): fobj[i].c /= 1.02
		f.py[:]=fobj[:]
		f.push()

		info, _ , _  = pogs(A, f, g, solver_state=solver_state,
			verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with solver state and 1.1x perturbed f(y)")

		f.pull()
		fobj = f.tolist()
		for i in xrange(0, m/2): fobj[i].c *= 1.1
		for i in xrange(m/2, m): fobj[i].c /= 1.1
		f.py[:]=fobj[:]
		f.push()

		info, _ , _  = pogs(A, f, g, solver_state=solver_state,
			verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with solver state and 1.2x perturbed f(y)")

		f.pull()
		fobj = f.tolist()
		for i in xrange(0, m/2): fobj[i].c *= 1.2
		for i in xrange(m/2, m): fobj[i].c /= 1.2
		f.py[:]=fobj[:]
		f.push()


		info, _ , _  = pogs(A, f, g, solver_state=solver_state,
			verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter

		PPRINT("WARM START with solver state and 2x perturbed f(y)")


		f.pull()
		fobj = f.tolist()
		for i in xrange(0, m/2): fobj[i].c *= 2
		for i in xrange(m/2, m): fobj[i].c /= 2
		f.py[:]=fobj[:]
		f.push()

		info, _ , _  = pogs(A, f, g, solver_state=solver_state,
			verbose = int(VERBOSE_TEST))
		var_assert(info, type=SolverInfo)
		assert info.err == 0
		assert info.converged or info.k == maxiter
		return True

	except:
		errors.append(format_exc())
		return False

def test_pogs(errors, *args, **kwargs):
	print "POGS CALL TESTING\n\n\n\n"
	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (None,None)
	A = np.load(kwargs['file']) if 'file' in kwargs else None

	success = warmstart_test(errors, m=m, n=n, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)
	if isinstance(A,np.ndarray): A=A.T
	success &= warmstart_test(errors, m=n, n=m, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)

	if success:
		print "...passed"
	else:
		print "...failed"
	return success