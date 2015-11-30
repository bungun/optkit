from optkit.types import Vector, Matrix
from optkit.kernels import *
from optkit.projector.direct import *
from optkit.blocksplitting import *
from optkit.blocksplitting.algorithms import *
import numpy as np
from operator import add



def blocksplitting_test(*size, **kwargs):

	HLINE  = reduce(add, ['-' for i in xrange(100)])


	if len(size)==2:
		(m,n)=int(size[0]),int(size[1])
	else:
		(m,n)=(30,20)

	VERBOSE_TEST = 'printall' in kwargs
		
	A = Matrix(np.random.rand(m,n))
	f = FunctionVector(m, h='Abs', b=1)
	g = FunctionVector(n, h='IndGe0')

	options = {}
	solver_state = None

	settings = SolverSettings(**options)
	info = SolverInfo()
	output = OutputVariables(m,n)

	A = SolverMatrix(A)
	z = ProblemVariables(m,n)
	d = z.de.y
	e = z.de.x


	print "\nPROBLEM SETUP METHODS"
	print "=====================\n"

	print "\nINPUT/OUTPUT"
	print "------------\n"

	print "\nPROBLEM DATA"
	print "A object: ", A
	if VERBOSE_TEST: print "A matrix: ", A.mat


	if VERBOSE_TEST:
		print "\FUNCTION VECTORS"
		print "f: ", f
		print "g: ", g

	if VERBOSE_TEST:
		print "\nPROBLEM VARIABLES"
		print "z: ", z

	if VERBOSE_TEST:
		print "\nEQUILIBRATION VARIABLES"
		print "d: ", d
		print "e: ", e

	print "\nSOLVER STRUCTS"
	print "settings: ", settings
	print "info: ", info 

	if VERBOSE_TEST:
		print "\nOUTPUT VARIABLES"
		print output


	print HLINE
	print "\nPRECONDITION, PROJECTOR, PROX OPERATOR"
	print "--------------------------------------\n"

	print "\nEQUILIBRATE"
	if not A.equilibrated: equilibrate(A, d, e)
	if VERBOSE_TEST:
		print "A: ", A
		print "d: ", d
		print "e: ", e
	else:
		print "...complete"

	print "\nPROJECTOR"
	Proj = DirectProjector(A.mat, normalize=True)		
	print "ProjA: ", Proj

	print "\nNORMALIZE A, Proj, d, e"
	if VERBOSE_TEST:
		print "BEFORE:"
		print "A object: ", A
		print "A matrix: ", A.mat
		print "d: ", d
		print "e: ", e
	normalize_system(A, Proj, d, e)
	if VERBOSE_TEST:
		print "AFTER:"
		print "A object: ", A
		print "A matrix: ", A.mat
		print "d: ", d
		print "e: ", e
	else:
		print "...complete"

	print "\nSCALE f, g"
	if VERBOSE_TEST:
		print "BEFORE"
		print "f: ", f
		print "g: ", g
	scale_functions(f,g,d,e)
	if VERBOSE_TEST:
		print "AFTER"
		print "f: ", f
		print "g: ", g
	else:
		print "...complete"


	print HLINE
	print "\nSCALED SYSTEM: PROJECTION TEST"
	x = Vector(np.random.rand(n))
	y = Vector(np.random.rand(m))
	x_out = Vector(n)
	y_out = Vector(m)

	print "RANDOM (x,y)"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x.py), np.linalg.norm(y.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y.py-A.mat.py.dot(x.py))

	Proj(x,y,x_out,y_out)

	print "PROJECT:"
	print "||x||_2, {} \t ||y||_2: {}".format(
		np.linalg.norm(x_out.py), np.linalg.norm(y_out.py))
	print "||Ax-y||_2:"
	print np.linalg.norm(y_out.py-A.mat.py.dot(x_out.py))


	print "\nPROXIMAL OPERATOR"
	Prox = prox_eval(f,g)		
	print "Prox: ", Prox	

	print HLINE
	print "\nVARIABLE MANIPULATION AND STORAGE"
	print "---------------------------------\n"

	print "\nVARIABLE INITIALIZATION"
	initialize_variables(A.mat, settings.rho, z, x.py, y.py)
	print "z primal", z.primal.vec
	print "z dual", z.dual.vec

	print "\nVARIABLE UNSCALING"
	unscale_output(0.5, z, output)		
	print "output vars:", output


	print "\nVARIABLE STORAGE"
	solver_state=SolverState(A,Proj,z)
	print solver_state

	print HLINE
	print HLINE
	print "\nPROBLEM SOLVE METHODS"
	print "=====================\n"

	print "\nACCESS SETTINGS"
	print "rho: ", settings.rho
	print "alpha: ", settings.alpha
	print "abs tol: ", settings.abstol
	print "rel tol: ", settings.reltol
	print "adpative rho: ", settings.adaptive
	print "max iter: ", settings.maxiter

	print HLINE
	print "\nMAKE OBJECTIVES, RESIDUALS, TOLERANCES"
	obj = Objectives()
	res = Residuals()
	eps = Tolerances(m,n, atol=settings.abstol, rtol=settings.reltol)
	print "Objectives: ", obj
	print "Residuals: ", res
	print "Tolerances: ", eps

	print HLINE
	print "\nMAKE ADAPTIVE RHO PARAMETERS"
	rhopar = AdaptiveRhoParameters()
	print "a.r. params: ", rhopar

	print HLINE
	print "\nITERATE: z_prev = z^k"
	z.prev.copy(z.primal)
	if VERBOSE_TEST:
		print z.prev
		print z.primal
	else:
		print "...complete"

	print HLINE
	print "\nPROX EVALUAION"
	if VERBOSE_TEST:
		print "BEFORE:"
		print z.primal12
	Prox(settings.rho,z)
	if VERBOSE_TEST:
		print "AFTER"
		print z.primal12
	else: 
		print "...complete"

	print HLINE
	print "\nPROJECTION"
	if VERBOSE_TEST:
		print "BEFORE:"
		print z.primal
	project_primal(Proj,z,alpha=settings.alpha)
	if VERBOSE_TEST:
		print "AFTER"
		print z.primal
	else:
		print "...complete"

	print HLINE
	print "\nDUAL UPDATE"
	if VERBOSE_TEST:
		print "BEFORE:"
		print "Z_TILDE"
		print z.dual
		print "Z_TILDE_1/2"
		print z.dual12
	update_dual(z,alpha=settings.alpha)
	if VERBOSE_TEST:
		print "AFTER"
		print "Z_TILDE"
		print z.dual
		print "Z_TILDE_1/2"
		print z.dual12 
	else:
		print "...complete"


	print HLINE
	print "\nCHECK CONVERGENCE:"
	converged = check_convergence(A,f,g,settings.rho,z,obj,res,eps,gapstop=settings.gapstop)	
	print "Converged? ", converged 
	

	print HLINE
	print "\nITERATION INFO:"
	print header_string()
	print iter_string(1, res, eps, obj)


	print HLINE
	print "\nADAPT RHO"
	print "Adaptive rho requested?", settings.adaptive
	print "rho before:", settings.rho
	if VERBOSE_TEST:
		print "Z_TILDE before:", z.dual
	if settings.adaptive:
		adapt_rho(z, rhopar, 0, settings, res, eps)
	print "rho after:", settings.rho
	if VERBOSE_TEST:
		print "Z_TILDE after:", z.dual


	print HLINE
	print "\nUPDATE INFO"
	print "info before: ", info
	info.update(rho=settings.rho,obj=obj.p, converged=converged,err=0,k=0)
	print "info after: ", info 



	print HLINE
	print HLINE
	print "\nPOGS INNER LOOP ROUTINE"
	print "-----------------------\n"
	settings.maxiter=2000
	print settings
	admm_loop(A.mat,Proj,Prox,z,settings,info, check_convergence(A.mat,f,g)) 	
	print info

	print HLINE
	print HLINE
	print "\nPOGS ROUTINE"
	print "------------\n"
	pogs(A.mat,f,g)



