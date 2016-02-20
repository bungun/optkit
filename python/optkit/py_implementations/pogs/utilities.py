from numpy import inf, sqrt, save
from numpy.linalg import norm
from toolz import curry
from os import path

class PogsKernels(object):
	def __init__(self, kernels, matrix_type, projector_type, pogs_types,
		equil_methods):

		self.call = kernels
		self.Matrix = matrix_type
		self.DirectProjector = projector_type
		self.SolverState = pogs_types.SolverState
		self.SolverMatrix = pogs_types.SolverMatrix
		self.ProblemVariables = pogs_types.ProblemVariables
		self.dense_l2_equilibration = equil_methods.dense_l2
		self.sinkhornknopp_equilibration = equil_methods.sinkhornknopp

		# Declare curried methods within __init__ for scoping reasons
		@curry
		def check_convergence(A,f,g,rho,admm_vars,obj,res,eps,
			gapstop=False, force_exact=False):
			z=admm_vars
			# compute gap, objective and tolerances.
			self.update_objective(f, g, rho, z, obj)
			self.update_tolerances(z, eps, obj)

		    # calculate residuals (exact only if necessary).
			exact=self.update_residuals(A, rho, z, obj, eps, res, force_exact=force_exact)

			# evaluate stopping criteria.
			return exact and \
			   res.p < eps.p and \
			   res.d < eps.d and \
			   (res.gap < eps.gap or not gapstop)

		self.check_convergence = check_convergence

		axpby_inplace = self.call['axpby_inplace']
		prox_eval_kernel = self.call['prox_eval']

		@curry
		def prox_eval(f, g, rho, admm_vars):
			z=admm_vars
			axpby_inplace(-1,z.dual.vec,1, z.primal.vec,z.temp.vec)
			prox_eval_kernel(f, rho, z.temp.y, z.primal12.y)
			prox_eval_kernel(g, rho, z.temp.x, z.primal12.x)

		self.prox_eval = prox_eval




	def overrelax(self, alpha, z12, z, z_out, overwrite=True):
		if alpha != 1:
			self.call['axpby'](alpha, z12.vec, int(not overwrite), z_out.vec)
			self.call['axpy'](1-alpha, z.vec, z_out.vec)
		else:
			z_out.copy_from(z12)

	def estimate_norm(self, A, *args):
		Warning("estimate_norm not implemented, returning norm=1")
		return 1.

	def equilibrate(self, A, d, e, method='sinkhorn'):
		if not isinstance(A, self.SolverMatrix):
			raise ValueError("Argument `A` must be of type `SolverMatrix`")

		m,n = A.shape
		if method == 'densel2':
			self.dense_l2_equilibration(A.orig,A.mat,d,e)
		else:
			self.sinkhornknopp_equilibration(A.orig,A.mat,d,e)
		A.equilibrated = True

	def normalize_system(self, A, d, e, normA=1.):

		(m,n)=A.shape
		if not A.normalized:
			normA = estimate_norm(A.mat)/sqrt(A.mat.mindim)
			call['div'](normA, A.mat)
			A.normalized = True

		factor = sqrt( self.call['nrm2'](d) / self.call['nrm2'](e) * \
			sqrt(n) / sqrt(m) )
		self.call['div'](factor*sqrt(normA), d)
		self.call['mul'](factor/sqrt(normA), e)


	def scale_functions(self, f, g, d, e):
		self.call['scale_function_vector'](f, d, mul=False)
		self.call['scale_function_vector'](g, e, mul=True)
		self.call['push_function_vector'](f, g)


	def update_objective(self, f, g, rho, admm_vars, obj):
		z=admm_vars
		obj.gap = rho*abs( self.call['dot'](z.primal12.vec, z.dual12.vec))
		obj.p = self.call['func_eval'](f, z.primal12.y) + \
			self.call['func_eval'](g, z.primal12.x)
		obj.d = obj.p - obj.gap

	def update_residuals(self, A, rho, admm_vars,
		obj, eps, res, force_exact=False):

		z=admm_vars
		self.call['axpby_inplace'](-1, z.primal.vec, 1, z.prev.vec, z.temp.vec)
		res.d = self.call['nrm2'](z.temp.vec)
		self.call['axpby_inplace'](-1, z.primal.vec, 1, z.primal12.vec, z.temp.vec)
		res.p = self.call['nrm2'](z.temp.vec)
		res.gap = obj.gap

		if (res.p < eps.p and res.d < eps.d) or force_exact:
			self.call['copy'](z.primal12.y, z.temp.y)
			self.call['gemv']('N',1,A,z.primal12.x,-1,z.temp.y)
			res.p = self.call['nrm2'](z.temp.y)
			if res.p < eps.p:
				self.call['copy'](z.dual12.x,z.temp.x)
				self.call['gemv']('T',1,A,z.dual12.y,1,z.temp.x)
				res.d = rho*self.call['nrm2'](z.temp.x)
				return True
		return False

	def iter_string(self, k, res, eps, obj):
		return str("   {}: {:.3e}, {:.3e}, {:.3e}, {:.3e}, "
					"{:.3e}, {:.3e}, {:.3e}".format(k,
					res.p, eps.d, res.d, eps.d,
					res.gap, eps.gap, obj.p))

	def header_string(self):
		return str("   #  res_pri     eps_pri   res_dual   eps_dual"
	           	   "   gap        eps_gap    objective\n");


	def update_tolerances(self, admm_vars, eps, obj):
		z=admm_vars
		eps.p = eps.atolm + eps.reltol * self.call['nrm2'](z.primal.y)
		eps.d = eps.atoln + eps.reltol * self.call['nrm2'](z.dual.x)
		eps.gap = eps.atolmn + eps.reltol * abs(obj.p)


	def adapt_rho(self, admm_vars, adaptive_rho_parameters, k, settings, res, eps):
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


	def check_warmstart_settings(self, **options):
		if 'no_warn' in options: return;
		args = []
		if 'x0' in options: args.append('x_guess')
		if 'nu0' in options: args.append('nu_guess')
		if 'rho' in options: args.append('rho')
		msg = str('\nWarning---recommended warm start configurations:\n'
			'Specify:'
			'\n\t(x_guess) [good],'
			'\n\t(x_guess, nu_guess) [better], or'
			'\n\t(x_guess, nu_guess, rho) [best].'
			'\nProvided:'
			'\n\t{}\n'.format(tuple(args)))
		if 'x0' in options and not ('nu0' in options and 'rho' in options):
			print "HERE1"
			print msg
		elif 'nu0' in options and not ('x0' in options and 'rho' in options):
			print "HERE2"
			print msg


	def initialize_variables(self, A, rho, admm_vars, x0, nu0):
		z=admm_vars
		if x0 is not None:
			self.call['copy'](x0, z.temp.x)
			self.call['div'](z.de.x, z.temp.x)
			self.call['gemv']('N', 1, A, z.temp.x, 0, z.temp.y)
			z.primal.copy_from(z.temp)
			z.primal12.copy_from(z.temp)

		if nu0 is not None:
			self.call['copy'](nu0, z.temp.y)
			self.call['div'](z.de.y, z.temp.y)
			self.call['gemv']('T', -1, A, z.temp.y, 0, z.temp.x)
			z.temp.mul(-1./rho)
			z.dual.copy_from(z.temp)

	def print_feasibility_conditions(self, A, admm_vars, msg=None):
		z=admm_vars
		print msg
		print "||x||", norm(z.primal.x.py)
		print "||y||", norm(z.primal.y.py)
		print "||Ax-y||", norm(A.mat.py.dot(z.primal.x.py)-z.primal.y.py)
		print "||nu||", norm(z.dual.y.py)
		print "||mu||", norm(z.dual.x.py)
		print "||A'nu+mu||", norm(A.mat.py.T.dot(z.dual.y.py)+z.dual.x.py)
		"--"
		print "||x12||", norm(z.primal12.y.py)
		print "||y12||", norm(z.primal12.x.py)
		print "||Ax12-y12||", norm(A.mat.py.dot(z.primal12.x.py)-z.primal12.y.py)
		print "||nu12||", norm(z.dual12.y.py)
		print "||mu12||", norm(z.dual12.x.py)
		print "||A'nu12+mu12||", norm(A.mat.py.T.dot(z.dual12.y.py)+z.dual12.x.py)

	def project_primal(self, Proj, admm_vars, alpha=None):
		z=admm_vars
		if alpha != None: self.overrelax(alpha, z.primal12, z.prev, z.temp)
		else: z.temp.copy_from(z.primal12)
		Proj(z.temp.x, z.temp.y, z.primal.x, z.primal.y)


	def update_dual(self, admm_vars, alpha=None):
		z=admm_vars

		z.dual12.copy_from(z.primal12)
		self.call['axpy'](-1, z.prev.vec, z.dual12.vec)
		self.call['axpy'](1, z.dual.vec, z.dual12.vec)

		if alpha != None: self.overrelax(alpha,z.primal12,z.prev,z.dual,
									overwrite=False)
		self.call['axpy'](-1, z.primal.vec, z.dual.vec)


	def unscale_output(self, rho, admm_vars, output_vars):
		z=admm_vars
		out=output_vars
		z.temp.copy_from(z.primal12)
		self.call['mul'](z.de.x, z.temp.x)
		self.call['div'](z.de.y, z.temp.y)
		z.temp.sync()
		out.x[:]=z.temp.x.py[:]
		out.y[:]=z.temp.y.py[:]


		z.temp.copy_from(z.dual12)
		z.temp.mul(-rho)
		self.call['div'](z.de.x, z.temp.x)
		self.call['mul'](z.de.y, z.temp.y)
		z.temp.sync()
		out.mu[:]=z.temp.x.py[:]
		out.nu[:]=z.temp.y.py[:]

	def save_solver_state(self, solver_state, output_file):
		if not isinstance(solver_state, self.SolverState):
			raise TypeError("first argument must be of type SolverState")
		if not isinstance(output_file, str):
			raise TypeError("second argument must be string")
		if not path.exists(path.dirname(output_file)):
			raise ValueError("second argument must start with a valid path")
		if path.basename(output_file) == 0:
			raise ValueError("filename not specified")
		if '.npz' not in path.basename(output_file):
			output_file += '.npz'
		proj_type = 'direct' if isinstance(solver_state.Proj,
			self.DirectProjector) else 'indirect'
		if proj_type == 'indirect':
			raise ValueError('save for indirect projector not implemented')
		else:
			L = solver_state.Proj.L
			A = solver_state.A.mat
			de = solver_state.z.de.vec
			primal = solver_state.z.primal.vec
			primal12 = solver_state.z.primal12.vec
			dual = solver_state.z.dual.vec
			dual12 = solver_state.z.dual12.vec
			prev = solver_state.z.prev.vec
			rho = solver_state.rho

			self.call['sync'](L, A, primal, primal12, dual, dual12,
				prev, de)


		save(output_file, L_equil=L.py, A_equil=A.py,
			de_equil=de.py, primal=primal.py,
			dual=dual.py, prev=prev.py,
			primal12=primal12.py, dual12=dual12.py,
			rho=rho, projector_type=proj_type)


	def load_solver_state(self, input_file):
		if not isinstance(input_file, str):
			raise TypeError("first argument must be string")
		if not 'npz' in input_file:
			raise ValueError("file ending in .npz expected")
		if not path.isfile(input_file):
			raise ValueError("input file cannot be opened")
		with np.load(input_file) as data:

			for item in ['primal', 'primal12', 'dual', 'dual12',
				'prev', 'de_equil', 'A_equil', 'L_equil',
				'rho', 'projector_type']:
				if not data.has_key(item):
					raise ValueError("file contents do not"
						"correspond to valid SolverState.")
			proj_type = data['projector_type']
			if proj_type == 'direct':
				A = self.SolverMatrix(self.Matrix(data['A_equil']))
				A.equilibrated=True
				A.normalized=True
				Proj = self.DirectProjector(A.mat,
					self.Matrix(data['L_equil']))
				Proj.normalized=True
				z = self.ProblemVariables(*A.shape)
				self.call['copy'](data['de_equil'], z.de.vec)
				self.call['copy'](data['prev'], z.prev.vec)
				self.call['copy'](data['primal'], z.primal.vec)
				self.call['copy'](data['primal12'], z.primal12.vec)
				self.call['copy'](data['dual'], z.dual.vec)
				self.call['copy'](data['dual12'], z.dual12.vec)
				solver_state = self.SolverState(A, Proj,
					z, data['rho'])

				return solver_state

			else:
				raise ValueError('load for indirect projector not implemented')