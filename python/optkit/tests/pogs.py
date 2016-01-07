import numpy as np
from numpy.linalg import norm
from optkit.types import ok_function_enums
from optkit.tests.defs import gen_test_defs
from optkit.utils.pyutils import println, pretty_print, var_assert

def warmstart_test(m=300, n=200, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

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



def pogs_test(m=300,n=200,A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

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
			(m,n) = A_in.shape
			A_in = A_in.astype(backend.lowtypes.FLOAT_CAST)
		else:
			A_in=None

	PRINT = println if VERBOSE_TEST else printvoid
	PPRINT = pretty_print if VERBOSE_TEST else printvoid
	verbose=2 if VERBOSE_TEST else 0

	mat_kind = "SKINNY MATRIX" if m >= n else "FAT MATRIX"

	PPRINT(mat_kind, '=')
	PRINT("m: {}, n: {}".format(m,n))

	A = Matrix(RAND_ARR(m,n)) if A_in is None else Matrix(A_in) 
	A_copy = np.copy(A.py)
	maxiter=2000
	reltol=1e-3

	for FUNCKEY in ok_function_enums.enum_dict.keys():
		print FUNCKEY

		f = FunctionVector(m, h=FUNCKEY, b=1)
		g = FunctionVector(n, h='IndGe0')
		info, output, solver_state = pogs(A, f, g,
			reltol=reltol, maxiter=maxiter, verbose=verbose)

		assert var_assert(solver_state, type=SolverState)
		assert var_assert(info, type=SolverInfo)
		assert var_assert(output, type=OutputVariables)
		assert info.converged or info.k==maxiter

		A_ = solver_state.A.mat.py 
		d = solver_state.z.de.y.py
		e = solver_state.z.de.x.py
		xrand = RAND_ARR(n)
		assert np.max(np.abs(d*A_copy.dot(e*xrand)-A_.dot(xrand)))

		res_p = np.linalg.norm(A_copy.dot(output.x)-output.y)
		res_d = np.linalg.norm(A_copy.T.dot(output.nu)+output.mu)

		if FUNCKEY not in  ('Logistic','Exp'):
			assert not info.converged or res_p <= TEST_EPS or res_p/np.linalg.norm(output.y) <= 10*reltol
		assert not info.converged or res_d <= TEST_EPS or res_d/np.linalg.norm(output.mu) <= 10*reltol


		PPRINT("PRIMAL & DUAL FEASIBILITY", '.')
		PRINT("||Ax-y||: ", res_p)
		PRINT("||A'nu+mu||: ", res_d)

		PPRINT("INFO:", '.')
		PRINT(info)
		
		PPRINT("OUTPUT:", '.')
		PRINT(output)

		PPRINT("SOLVER STATE:", '.')
		PRINT(solver_state)



def test_pogs(*args, **kwargs):
	print "POGS CALL TESTING\n\n\n\n"
	verbose = '--verbose' in args
	floatbits = 32 if 'float' in args else 64
	(m,n)=kwargs['shape'] if 'shape' in kwargs else (None,None)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	# pogs_test(m=m,n=n,A_in=A,VERBOSE_TEST=verbose)
	# if isinstance(A,np.ndarray): A=A.T
	# pogs_test(m=n,n=m,A_in=A,VERBOSE_TEST=verbose)

	warmstart_test(m=m, n=n, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)
	if isinstance(A,np.ndarray): A=A.T
	warmstart_test(m=n, n=m, A_in=A, VERBOSE_TEST=verbose,
		gpu='gpu' in args, floatbits=floatbits)
	print "...passed"
	return True