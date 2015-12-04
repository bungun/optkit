from optkit.types import Vector
from optkit.kernels import *
from optkit.projector.direct import *
from optkit.blocksplitting.types import *
from optkit.equilibration.methods import *
from numpy import inf, sqrt
from toolz import curry
from numpy.linalg import norm
import sys

def blockdot(bv1,bv2):
	return dot(bv1.vec,bv2.vec)

def blockcopy(bv_src,bv_dest):
	return copy(bv_src.vec,bv_dest.vec)

def overrelax(alpha, z12, z, z_out, overwrite=True):
	if alpha != 1:
		axpby(alpha, z12.vec, int(not overwrite), z_out.vec)
		axpy(1-alpha, z.vec, z_out.vec)
	else:
		copy(z12.vec, z_out.vec)


def estimate_norm(A, *args):
	Warning("estimate_norm not implemented, returning norm=1")
	return 1.

def equilibrate(A, d, e, method='sinkhorn'):
	if not isinstance(A,SolverMatrix):
		raise ValueError("Argument `A` must be of type `SolverMatrix`")


	m,n = A.shape
	if method=='densel2':
		dense_l2_equilibration(A.orig,A.mat,d,e)
	else:
		sinkhornknopp_equilibration(A.orig,A.mat,d,e)
	A.equilibrated = True

def normalize_system(A, d, e, normA=1.):

	(m,n)=A.shape
	if not A.normalized:
		normA = estimate_norm(A.mat)/sqrt(A.mat.mindim)
		div(normA, A.mat)
		A.normalized = True

	factor = sqrt( nrm2(d)/nrm2(e) * sqrt(n)/sqrt(m))
	div(factor*sqrt(normA), d)
	mul(factor/sqrt(normA), e)


def scale_functions(f, g, d, e):
	scale_function_vector(f,d,mul=False)
	scale_function_vector(g,e,mul=True)
	push_function_vector(f,g)


def update_objective(f, g, rho, admm_vars, obj):
	z=admm_vars
	obj.gap = rho*abs(dot(z.primal12.vec,z.dual12.vec))
	obj.p = eval(f, z.primal12.y) + eval(g, z.primal12.x)
	obj.d = obj.p - obj.gap

def update_residuals(A, rho, admm_vars, obj, eps, res):
	z=admm_vars
	axpby_inplace(-1, z.primal.vec, 1, z.prev.vec, z.temp.vec)
	res.d = nrm2(z.temp.vec)
	axpby_inplace(-1, z.primal.vec, 1, z.primal12.vec, z.temp.vec)
	res.p = nrm2(z.temp.vec)
	res.gap = obj.gap

	if res.p < eps.p and res.d < eps.d:
		copy(z.primal12.y, z.temp.y)
		gemv('N',1,A,z.primal12.x,-1,z.temp.y)
		res.p = nrm2(z.temp.y)
		if res.p < eps.p:
			copy(z.dual12.x,z.temp.x)
			gemv('T',1,A,z.dual12.y,1,z.temp.x)
			res.d = rho*nrm2(z.temp.x)
			return True
	return False

def iter_string(k, res, eps, obj):
	return str("   {}: {:.3e}, {:.3e}, {:.3e}, {:.3e}, "
				"{:.3e}, {:.3e}, {:.3e}".format(k, 
				res.p, eps.d, res.d, eps.d, 
				res.gap, eps.gap, obj.p))

def header_string():
	return str("   #  res_pri     eps_pri   res_dual   eps_dual"
           	   "   gap        eps_gap    objective\n");


def update_tolerances(admm_vars, eps, obj):
	z=admm_vars
	eps.p = eps.atolm + eps.reltol * nrm2(z.primal.y)
	eps.d = eps.atoln + eps.reltol * nrm2(z.dual.x)
	eps.gap = eps.atolmn + eps.reltol * abs(obj.p)

@curry
def check_convergence(A,f,g,rho,admm_vars,obj,res,eps,gapstop=False):
	z=admm_vars
	# compute gap, objective and tolerances.
	update_objective(f, g, rho, z, obj)
	update_tolerances(z, eps, obj)

    # calculate residuals (exact only if necessary).
	exact=update_residuals(A, rho, z, obj, eps, res)

	# evaluate stopping criteria.
	return exact and \
	   res.p < eps.p and \
	   res.d < eps.d and \
	   (res.gap < eps.gap or not gapstop)

def adapt_rho(admm_vars, adaptive_rho_parameters, k, settings, res, eps):
	params = adaptive_rho_parameters

	if res.d < params.xi*eps.d and res.p < params.xi*eps.p:
		params.xi *= params.KAPPA
	elif res.d < params.xi * eps.d and params.TAU*k > params.l \
	   and res.p > params.xi * eps.p:
		if settings.rho < params.RHOMAX:
			settings.rho *= params.delta
			if settings.verbose > 2:
				print "+RHO", settings.rho
			admm_vars.dual.div(params.delta)
			params.delta = min(params.delta * params.GAMMA, params.DELTAMAX)
			params.u = k
	elif res.p < params.xi * eps.p and params.TAU*k > params.u \
		and res.d > params.xi * eps.d:
		if settings.rho > params.RHOMIN:
			settings.rho /= params.delta
			if settings.verbose > 2:
				print "-RHO", settings.rho
			admm_vars.dual.mul(params.delta)
			params.delta=min(params.delta * params.GAMMA, params.DELTAMAX)
			params.l = k
	else:
		params.delta = max(params.delta / params.GAMMA, params.DELTAMIN)




def initialize_variables(A, rho, admm_vars, x0, nu0):
	z=admm_vars
	if x0 is not None:
		z.temp.x.py[:]=x0[:]
		sync(z.temp.x, python_to_C=1)
		gemv('N', 1, A, z.temp.x, 0, z.temp.y)
		z.primal.copy_from(z.temp)
	if nu0 is not None:
		z.temp.y.py[:]=nu0[:]
		sync(z.temp.y, python_to_C=1)
		gemv('T', -1, A, z.temp.y, 0, z.temp.x)
		mul(-1./rho, z.temp.vec)
		z.dual.copy_from(z.temp)



@curry 
def prox_eval(f,g,rho,admm_vars):
	z=admm_vars
	axpby_inplace(-1,z.dual.vec,1, z.primal.vec,z.temp.vec)
	prox(f, rho, z.temp.y, z.primal12.y)
	prox(g, rho, z.temp.x, z.primal12.x)

@curry		
def project_primal(Proj, admm_vars, alpha=None):
	z=admm_vars
	if alpha != None: overrelax(alpha, z.primal12, z.prev, z.temp)
	else: z.temp.copy_from(z.primal12)
	Proj(z.temp.x, z.temp.y, z.primal.x, z.primal.y)


def update_dual(admm_vars, alpha=None):
	z=admm_vars

	z.dual12.copy_from(z.primal12)
	axpy(-1, z.prev.vec, z.dual12.vec)
	axpy(1, z.dual.vec, z.dual12.vec)
	
	if alpha != None: overrelax(alpha,z.primal12,z.prev,z.dual,
								overwrite=False)
	axpy(-1, z.primal.vec, z.dual.vec)


def unscale_output(rho, admm_vars, output_vars):
	z=admm_vars
	out=output_vars
	z.temp.copy_from(z.primal12)
	mul(z.de.x, z.temp.x)
	div(z.de.y, z.temp.y)
	sync(z.temp.vec)
	out.x[:]=z.temp.x.py[:]
	out.y[:]=z.temp.y.py[:]


	z.temp.copy_from(z.dual12)
	mul(-rho, z.temp.vec)
	div(z.de.x, z.temp.x)
	mul(z.de.y, z.temp.y)
	sync(z.temp.vec)
	out.mu[:]=z.temp.x.py[:]
	out.nu[:]=z.temp.y.py[:]

def store_final_iteration(admm_vars):
	admm_vars.primal.copy_from(admm_vars.primal12)
	
