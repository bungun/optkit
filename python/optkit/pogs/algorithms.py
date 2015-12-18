from optkit.pogs.types import PogsTypes
from optkit.pogs.utilities import PogsKernels
from sys import version_info

version = version_info[0]+0.1*version_info[1]
if version >= 3.3:
	from time import process_time as timer
else:
	from time import clock as timer

version_info[0]+0.1*version_info[1]

class POGSDirectSolver(object):
	def __init__(self, backend, kernels, vector_type, matrix_type, 
		functionvector_type, projector_type, equilibration_methods):

		self.FunctionVector = functionvector_type
		self.DirectProjector = projector_type
		self.types = PogsTypes(backend, kernels, vector_type, matrix_type, 
			projector_type)
		self.utils = PogsKernels(kernels, matrix_type, 
			projector_type, self.types, equilibration_methods)
		self.load_solver_state = self.utils.load_solver_state
		self.save_solver_state = self.utils.save_solver_state

	def admm_loop(self, A, Proj, Prox, admm_vars, settings, info, stopping_conditions):
		#-----------------------------------------------
		# get handles to all necessary function calls
		project_primal = self.utils.project_primal
		update_dual = self.utils.update_dual
		adapt_rho = self.utils.adapt_rho
		#-----------------------------------------------


		#-----------------------------------------------
		# setup solver parameters
		err=0
		alpha=settings.alpha
		PRINT_ITER = 10000/(10**settings.verbose)
		if settings.verbose == 0: 
			PRINT_ITER = settings.maxiter * 2

		(m,n)=(A.size1,A.size2)
		obj = self.types.Objectives() 							
		res = self.types.Residuals() 							
		eps = self.types.Tolerances(m,n,atol=settings.abstol,rtol=settings.reltol) 			
		rho_params = self.types.AdaptiveRhoParameters() 		
		
		converged = False
		k = 0
		#-----------------------------------------------

		#-----------------------------------------------
		# solver loop

	  	if not err and settings.verbose > 0:	# signal start of execution.
			print self.utils.header_string()

		z=admm_vars
		while not (err or converged or k==settings.maxiter):
			k+=1

			z.prev.copy_from(z.primal)			# z 	= z.1
			Prox(settings.rho,z)				# z.12 	= prox(z - zt)
			project_primal(Proj, z, 
					alpha=alpha)				# z.1 	= proj(z.12 + zt)


			update_dual(z,alpha=alpha)			# zt.1 	= zt.1 + z.12 - z.1
												# zt.12 = zt + z.12 - z

			converged = stopping_conditions(	# stopping criteria
				settings.rho,z, obj,res,eps,
				gapstop=settings.gapstop,
				force_exact=(settings.resume and k==1))	

			if settings.verbose > 0 and (k % PRINT_ITER == 0 or converged):								
				print(self.utils.iter_string(k,res,eps,obj))

			if settings.adaptive: 				# adapt rho, rescale dual
				adapt_rho(z, rho_params,k,settings,res,eps)
		#-----------------------------------------------

		#-----------------------------------------------
		# check loop exit, update solver info
		if not converged and k==settings.maxiter:
			print "reached max iter={}".format(settings.maxiter)

		info.update(rho=settings.rho,obj=obj.p,converged=converged,err=err, k=k)
		#-----------------------------------------------


	def __call__(self, A,f,g, solver_state=None, **options):

		#-----------------------------------------------
		# get handles to all necessary types
		FunctionVector = self.FunctionVector	
		DirectProjector = self.DirectProjector	
		ProblemVariables = self.types.ProblemVariables
		SolverMatrix = self.types.SolverMatrix
		SolverSettings = self.types.SolverSettings
		SolverInfo = self.types.SolverInfo
		SolverState = self.types.SolverState
		OutputVariables = self.types.OutputVariables

		# get handles to all necessary function calls
		equilibrate = self.utils.equilibrate
		normalize_system = self.utils.normalize_system
		scale_functions = self.utils.scale_functions
		check_convergence = self.utils.check_convergence
		initialize_variables = self.utils.initialize_variables
		print_feasibility_conditions = self.utils.print_feasibility_conditions
		prox_eval = self.utils.prox_eval
		unscale_output = self.utils.unscale_output

		#-----------------------------------------------

		#-----------------------------------------------
		# start setup timer
		t_start = timer()						

		# get & check problem dimensions
		(m,n) = A.shape
		try:
			assert f.size == m
			assert g.size == n
		except AssertionError:
			print "f, g must be sized compatibly with problem matrix A."
			raise

		# scope-specific copies of function vector
		f = FunctionVector(m, f=f)
		g = FunctionVector(n, f=g)

		# create solver variables
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

		# matrix equilibration: A_equil = D*A*E
		d = z.de.y
		e = z.de.x
		if not A.equilibrated: equilibrate(A, d, e)

		# set up projector; normalize A; adjust d, e & projector accordingly
		if solver_state is not None: 
			Proj = solver_state.Proj
		else: 
			Proj = DirectProjector(A.mat, normalize=True)		
			A.normalized=True
			normalize_system(A, d, e, Proj.normA)



		# scale objectives to account for d, e
		scale_functions(f,g,d,e)				

		# set up proximal operators and stopping criteria
		Prox = prox_eval(f,g) 						
		conditions = check_convergence(A.mat,f,g)


		# get warm start variables, if provided
		if settings.warmstart:					
			x0 = options['x0'] if 'x0' in options else None
			nu0 = options['nu0'] if 'nu0' in options else None
			if x0 is not None or nu0 is not None:
				initialize_variables(A.mat, settings.rho, z, x0, nu0)

		# stop setup timer
		info.setup_time = timer() - t_start		
		#-----------------------------------------------	

		#-----------------------------------------------
		# solve

		t_start = timer() 						

		self.admm_loop(A.mat, Proj, Prox, z, 
			settings, info, conditions) 		
		
		info.solve_time = timer() - t_start		
		
		#-----------------------------------------------

		#-----------------------------------------------
		# retrieve output

		unscale_output(info.rho, z, output)		

		if solver_state is None: 
			solver_state = SolverState(A, z, Proj, info.rho)
		else:
			solver_state.rho = info.rho

		if settings.verbose > 0: info.print_status(A.orig, output)
		#-----------------------------------------------


		return info, output, solver_state