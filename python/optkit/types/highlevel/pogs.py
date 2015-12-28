from optkit.types import ok_enums
from numpy import zeros, ndarray


class HighLevelPogsTypes(object):
	def __init__(self, backend, function_vector_type):
		ndarray_pointer = backend.lowtypes.ndarray_pointer
		PogsSettings = backend.pogs.pogs_settings
		PogsInfo = backend.pogs.pogs_info
		PogsOutput = backend.pogs.pogs_output
		pogslib = backend.pogs
		FunctionVector = function_vector_type

		# TODO: set direct/indirect, e.g., 
		# pogslib_direct = backend.pogs_direct
		# pogslib_indirect = backend.pogs_indirect


		class SolverSettings(object):
			def __init__(self, **options):
				self.c = PogsSettings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None)
				pogslib.set_default_settings(self.c)
				self.update(**options)

			def update(self, **options):
				if 'alpha' in options: self.c.alpha = options['alpha']
				if 'rho' in options: self.c.rho = options['rho']
				if 'abstol' in options: self.c.abstol = options['abstol']
				if 'reltol' in options: self.c.reltol = options['reltol']
				if 'maxiter' in options: self.c.maxiter = options['maxiter']
				if 'verbose' in options: self.c.verbose = options['verbose']
				if 'adaptiverho' in options: self.c.adaptiverho = options['adaptiverho']
				if 'gapstop' in options: self.c.gapstop = options['gapstop']
				if 'resume' in options: self.c.resume = options['resume']
				if 'x0' in options: self.c.x0 = ndarray_pointer(options['x0'])
				if 'nu0' in options: self.c.nu0 = ndarray_pointer(options['nu0'])

			def __str__(self):
				return str(
					"alpha: {}\n".format(self.c.alpha) + 
					"rho: {}\n".format(self.c.rho) +
					"abstol: {}\n".format(self.c.abstol) +
					"reltol: {}\n".format(self.c.reltol) +
					"maxiter: {}\n".format(self.c.maxiter) +
					"verbose: {}\n".format(self.c.verbose) +
					"adaptiverho: {}\n".format(self.c.adaptiverho) +
					"gapstop: {}\n".format(self.c.gapstop) +
					"warmstart: {}\n".format(self.c.warmstart) +
					"resume: {}\n".format(self.c.resume) +
					"x0: {}\n".format(self.c.x0) +
					"nu0: {}\n".format(self.c.nu0))
		
		self.SolverSettings = SolverSettings

		class SolverInfo(object):
			def __init__(self):
				self.c = PogsInfo(0, 0, 0, 0, 0, 0, 0)

			def __str__(self):
				return str(
					"error: {}\n".format(self.c.err) + 
					"converged: {}\n".format(self.c.converged) +
					"iterations: {}\n".format(self.c.k) +
					"objective: {}\n".format(self.c.obj) +
					"rho: {}\n".format(self.c.rho) +
					"setup time: {}\n".format(self.c.setup_time) +
					"solve time: {}\n".format(self.c.solve_time))

		self.SolverInfo = SolverInfo

		class SolverOutput(object):
			def __init__(self, m, n):
				self.x = zeros(n)
				self.y = zeros(m)
				self.mu = zeros(n)
				self.nu = zeros(m)
				self.c = PogsOutput(ndarray_pointer(self.x), ndarray_pointer(self.y),
					ndarray_pointer(self.mu), ndarray_pointer(self.nu))

			def __str__(self):
				return str("x:\n{}\ny:\n{}\nmu:\n{}\nnu:\n{}\n".format(
					str(self.x), str(self.y), 
					str(self.mu), str(self.nu)))

		self.SolverOutput = SolverOutput

		class Solver(object):
			def __init__(self, A, **options):
				try:
					assert isinstance(A, ndarray)
					assert len(A.shape) == 2
				except:
					raise TypeError('input must be a 2-d numpy ndarray')


				self.A = A 
				self.shape = (self.m, self.n) = (m, n) = A.shape
				self.layout = layout = ok_enums.CblasRowMajor if \
					A.flags.c_contiguous else ok_enums.CblasColMajor
				self.c_solver = pogslib.pogs_init(ndarray_pointer(A), m , n,
					layout, pogslib.enums.EquilSinkhorn)
				self.settings = SolverSettings()
				self.info = SolverInfo()
				self.output = SolverOutput(m, n)
				self.settings.update(**options)
				self.first_run = True

			def solve(self, f, g, **options):
				if not (isinstance(f, FunctionVector) and \
					isinstance(f, FunctionVector)):

					raise TypeError(
						'inputs f, g must be of type optkit.FunctionVector'
						'\nprovided: {}, {}'.format(type(f), type(g)))

				if not (f.size == self.m and g.size == self.n):
					raise ValueError(
						'inputs f, g not compatibly sized with solver'
						'\nsolver dimensions ({}, {})\n provided: ({}{})'.format(
							self.m, self.n, f.size, g.size))


				# TODO : logic around resume, warmstart, rho input

				self.settings.update(**options)
				pogslib.pogs_solve(self.c_solver, f.c, g.c, 
					self.settings.c, self.info.c, self.output.c)
				self.first_run = False

			def load(self):
				pass

			def save(self):
				pass
					
			def __del__(self):
				pogslib.pogs_finish(self.c_solver)

		self.Solver = Solver







