from optkit.types import Vector
from optkit.kernels import *
from optkit.projector.direct import *
from optkit.blocksplitting.types import *
from optkit.blocksplitting.utilities import *
from numpy import inf 
from numpy.linalg import norm
from time import time

def admm_loop(A, Proj, Prox, admm_vars, settings, info, stopping_conditions):
	err=0
	alpha=settings.alpha
	PRINT_ITER = 10000/(10**settings.verbose)
	if settings.verbose == 0: 
		PRINT_ITER = settings.maxiter * 2

	(m,n)=(A.size1,A.size2)
	obj = Objectives() 							
	res = Residuals() 							
	eps = Tolerances(m,n,atol=settings.abstol,rtol=settings.reltol) 			
	rho_params = AdaptiveRhoParameters() 		
	
	converged = False
	k = 0

  	if not err and settings.verbose > 0:	# signal start of execution.
		print header_string()

	z=admm_vars
	while not (err or converged or k==settings.maxiter):
		k+=1

		z.prev.copy_from(z.primal)			# z 	= z.1
		Prox(settings.rho,z)				# z.12 	= prox(z - zt)
		project_primal(Proj,z,
						alpha=alpha)		# z.1 	= proj(z.12 + zt)


		update_dual(z,alpha=alpha)			# zt.1 	= zt.1 + z.12 - z.1
											# zt.12 = zt + z.12 - z

		converged = stopping_conditions(	# stopping criteria
			settings.rho,z, obj,res,eps,
			gapstop=settings.gapstop,
			force_exact=(settings.resume and k==1))	

		if settings.verbose > 0 and (k % PRINT_ITER == 0 or converged):								
			print(iter_string(k,res,eps,obj))

		if settings.adaptive: 				# adapt rho, rescale dual
			adapt_rho(z, rho_params,k,settings,res,eps)

	if not converged and k==settings.maxiter:
		print "reached max iter={}".format(settings.maxiter)

	info.update(rho=settings.rho,obj=obj.p,converged=converged,err=err, k=k)



def pogs(A,f,g, solver_state=None, **options):
	err = 0

	t_start = time()							# start setup timer

	(m,n) = A.shape
	assert f.size == m
	assert g.size == n

	# scope-specific copies of function vector
	f = FunctionVector(m, f=f)
	g = FunctionVector(n, f=g)

	settings = SolverSettings(**options)
	info = SolverInfo()
	output = OutputVariables(m,n)

	if solver_state is not None:
		A = solver_state.A
		z = solver_state.z
		if not 'rho' in options and solver_state.rho is not None:
			settings.update(rho=solver_state.rho,resume=True) 

		if settings.debug:
			print_feasibility_conditions(A, z, msg="REBOOT CONDITIONS")



	else:
		A = SolverMatrix(A)
		z = ProblemVariables(m,n)

	d = z.de.y
	e = z.de.x

	if not A.equilibrated: equilibrate(A, d, e)

	if solver_state is not None: 
		Proj = solver_state.Proj
	else: 
		Proj = DirectProjector(A.mat, normalize=True)		
		A.normalized=True
		normalize_system(A, d, e, Proj.normA)

	scale_functions(f,g,d,e)				

	Prox = prox_eval(f,g) 						
	conditions = check_convergence(A.mat,f,g)

	x0=nu0=None
	if settings.warmstart:					# get initial guess
		# check_warmstart_settings(**options)
		x0 = options['x0'] if 'x0' in options else None
		nu0 = options['nu0'] if 'nu0' in options else None
		if x0 is not None or nu0 is not None:
			initialize_variables(A.mat, settings.rho, z, x0, nu0)


	info.setup_time = t_start - time()		# stop setup timer
	
	t_start = time() 						# start solver timer

	admm_loop(A.mat,Proj,Prox,z,settings,info,conditions) # execute ADMM loop
	
	info.solve_time = t_start - time() 		# stop solver timer
									
	unscale_output(info.rho, z, output)		# unscale system


	# TODO: update solver state if it exists already
	if solver_state is None: solver_state = SolverState(A,z,Proj,info.rho)

	if settings.verbose > 0: info.print_status


	return info, output, solver_state


