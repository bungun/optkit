from optkit.types import Vector
from optkit.kernels import *
from optkit.projector.direct import *
from optkit.blocksplitting.types import *
from optkit.blocksplitting.utilities import *
from numpy import inf 


def admm_loop(A, Proj, Prox, admm_vars, settings, info, stopping_conditions):
	err=0
	alpha=settings.alpha
	PRINT_ITER = 1000/(10**settings.verbose)

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

		z.prev.copy(z.primal)				# z 	= z.1
		Prox(settings.rho,z)				# z.12 	= prox(z - zt)
		project_primal(Proj,z,
						alpha=alpha)		# z.1 	= proj(z.12 + zt)



		update_dual(z,alpha=alpha)			# zt.1 	= zt.1 + z.12 - z.1
											# zt.12 = zt + z.12 - z

		converged = stopping_conditions(	# stopping criteria
			settings.rho,z, obj,res,eps,
			gapstop=settings.gapstop)	

		if settings.verbose > 0 and (k % PRINT_ITER == 0 or converged):								
			print(iter_string(k,res,eps,obj))

		if settings.adaptive: 				# adapt rho, rescale dual
			adapt_rho(z, rho_params,k,settings,res,eps)

	if not converged and k==settings.maxiter:
		print "reached max iter={}".format(settings.maxiter)

	print k
	print settings.rho
	info.update(rho=settings.rho,obj=obj.p,converged=converged,err=err, k=k)



def pogs(A,f,g, solver_state=None, **options):
	err = 0

	(m,n) = A.shape
	assert f.size == m
	assert g.size == n

	settings = SolverSettings(**options)
	info = SolverInfo()
	output = OutputVariables(m,n)

	if solver_state is not None:
		A = solver_state.A
		z = solver_state.z
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

	normalize_system(A, Proj, d, e)
	scale_functions(f,g,d,e)				

	Prox = prox_eval(f,g) 						
	conditions = check_convergence(A.mat,f,g)

	x0=nu0=None
	if settings.warmstart:					# get initial guess
		x0 = options['x0'] if 'x0' in options else None
		nu0 = options['nu0'] if 'nu0' in options else None
		initialize_variables(A.mat, settings.rho, z, x0, nu0)

	admm_loop(A.mat,Proj,Prox,z,settings,info,conditions) # execute ADMM loop

									
	unscale_output(info.rho, z, output)		# unscale system


	# TODO: update solver state if it exists already
	if solver_state is None: solver_state = SolverState(A,z,Proj)

	return info, output, solver_state


