from optkit.types import ok_enums
from numpy import zeros, ndarray, savez, load as np_load
from ctypes import c_void_p
from os import path

class HighLevelPogsTypes(object):
	def __init__(self, backend, function_vector_type):
		FLOAT_CAST = backend.lowtypes.FLOAT_CAST
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
				self.x = zeros(n, dtype=FLOAT_CAST)
				self.y = zeros(m, dtype=FLOAT_CAST)
				self.mu = zeros(n, dtype=FLOAT_CAST)
				self.nu = zeros(m, dtype=FLOAT_CAST)
				self.mu1 = zeros(n, dtype=FLOAT_CAST)
				self.nu1 = zeros(m, dtype=FLOAT_CAST)
				self.c = PogsOutput(ndarray_pointer(self.x), ndarray_pointer(self.y),
					ndarray_pointer(self.mu), ndarray_pointer(self.nu))

			def __str__(self):
				return str("x:\n{}\ny:\n{}\nmu:\n{}\nnu:\n{}\n".format(
					str(self.x), str(self.y), 
					str(self.mu), str(self.nu),
					str(self.mu1), str(self.nu1)))

		self.SolverOutput = SolverOutput

		class Solver(object):
			def __init__(self, A, *args, **options):
				try:
					assert isinstance(A, ndarray)
					assert len(A.shape) == 2
				except:
					raise TypeError('input must be a 2-d numpy ndarray')

				self.A = FLOAT_CAST(A) 
				self.shape = (self.m, self.n) = (m, n) = A.shape
				self.layout = layout = ok_enums.CblasRowMajor if \
					A.flags.c_contiguous else ok_enums.CblasColMajor

				if 'no_init' not in args:
					self.c_solver = pogslib.pogs_init(
						ndarray_pointer(self.A), m , n,
						layout, pogslib.enums.EquilSinkhorn)
				else:
					self.c_solver = None

				self.settings = SolverSettings()
				self.info = SolverInfo()
				self.output = SolverOutput(m, n)
				self.settings.update(**options)
				self.first_run = True



			def solve(self, f, g, **options):
				if self.c_solver is None: 
					Warning("No solver intialized, solve() call invalid")
					return

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

			def load(self, directory, name):
				filename = path.join(directory, name)
				if not '.npz' in name: 
					filename += '.npz'

				err = 0
				try:	
					data = np_load(filename)
				except:
					data = {}


				if 'A_equil' in data:
					A_equil = FLOAT_CAST(data['A_equil'])
				elif path.exists(path.join(directory, 'A_equil.npy')):
					A_equil = FLOAT_CAST(np_load(path.join(directory, 'A_equil.npy')))
				else: 
					err = 1


				if 'LLT' in data:
					LLT = FLOAT_CAST(data['LLT'])
					LLT_ptr = ndarray_pointer(LLT)
				elif path.exists(path.join(directory, 'LLT.npy')):
					LLT = FLOAT_CAST(np_load(path.join(directory, 'LLT.npy')))
					LLT_ptr = ndarray_pointer(LLT)
				else:
					if pogslib.direct:
						err = 1
					else:
						LLT_ptr = c_void_p()


				if 'd' in data:
					d = FLOAT_CAST(data['d'])
				elif path.exists(path.join(directory, 'd.npy')):
					d = FLOAT_CAST(np_load(path.join(directory, 'd.npy')))
				else:
					err = 1

				if 'e' in data:
					e = FLOAT_CAST(data['e'])
				elif path.exists(path.join(directory, 'e.npy')):
					e = FLOAT_CAST(np_load(path.join(directory, 'e.npy')))
				else:
					err = 1

				if err:
					snippet = '`LLT`, ' if pogslib.direct else ''
					ValueError('Minimal requirements to load solver '
						'not met. Specified file must contain '
						'at least one .npz file with entries `A_equil`, '
						'{}`d`, and `e`, or the specified folder must'
						'contain .npy files of the same names.'.format(
							snippet))

				if 'z' in data:
					z = FLOAT_CAST(data['z'])
				else:
					z = zeros(self.m + self.n, dtype=FLOAT_CAST)

				if 'z12' in data:
					z12 = FLOAT_CAST(data['z12'])
				else:
					z12 = zeros(self.m + self.n, dtype=FLOAT_CAST)

				if 'zt' in data:
					zt = FLOAT_CAST(data['zt'])
				else:	
					zt = zeros(self.m + self.n, dtype=FLOAT_CAST)

				if 'zt12' in data:
					zt12 = FLOAT_CAST(data['zt12'])
				else:
					zt12 = zeros(self.m + self.n, dtype=FLOAT_CAST)

				if 'zprev' in data:
					zprev = FLOAT_CAST(data['zprev'])
				else:
					zprev = zeros(self.m + self.n, dtype=FLOAT_CAST)

				if 'rho' in data:
					rho = FLOAT_CAST(data['rho'])
				else:
					rho = 1.

				order = ok_enums.CblasRowMajor if A_equil.flags.c_contiguous \
					else ok_enums.CblasColMajor

				if self.c_solver is not None:
					pogslib.pogs_finish(self.c_solver)

				self.c_solver = pogslib.pogs_load_solver(
					ndarray_pointer(A_equil), 
					LLT_ptr, ndarray_pointer(d), 
					ndarray_pointer(e), ndarray_pointer(z),
					ndarray_pointer(z12), ndarray_pointer(zt),
					ndarray_pointer(zt12), ndarray_pointer(zprev), 
					rho, self.m, self.n, order)

			def save(self, directory, name, 
				save_equil=True, save_factorization=True):
				if self.c_solver is None: 
					Warning("No solver intialized, save() call invalid")
					return


				filename = path.join(directory, name)
				if '.npz' not in name: filename += '.npz'

				if not path.exists(directory):
					Warning("specified directory does not exist")
					return 
				if path.exists(filename):
					Warning("specified filepath already exists"
						"and would be overwritten, aborting.")
					return

				mindim = min(self.m, self.n)
				A_equil = zeros((self.m, self.n), dtype=FLOAT_CAST)
				if pogslib.direct:
					LLT = zeros((mindim, mindim), dtype=FLOAT_CAST)
					LLT_ptr = ndarray_pointer(LLT)
				else:
					LLT = c_void_p()
					LLT_ptr = LLT
				order = ok_enums.CblasRowMajor if A_equil.flags.c_contiguous \
					else ok_enums.CblasRowMajor

				d = zeros(self.m, dtype=FLOAT_CAST)
				e = zeros(self.n, dtype=FLOAT_CAST)
				z = zeros(self.m + self.n, dtype=FLOAT_CAST)
				z12 = zeros(self.m + self.n, dtype=FLOAT_CAST)
				zt = zeros(self.m + self.n, dtype=FLOAT_CAST)
				zt12 = zeros(self.m + self.n, dtype=FLOAT_CAST)
				zprev = zeros(self.m + self.n, dtype=FLOAT_CAST)
				rho = zeros(1, dtype=FLOAT_CAST)


				pogslib.pogs_extract_solver(self.c_solver, 
					ndarray_pointer(A_equil), 
					LLT_ptr, ndarray_pointer(d), 
					ndarray_pointer(e), ndarray_pointer(z),
					ndarray_pointer(z12), ndarray_pointer(zt),
					ndarray_pointer(zt12), ndarray_pointer(zprev), 
					ndarray_pointer(rho), order)

				if isinstance(LLT, ndarray) and save_factorization:
					savez(filename, 
						A_equil=A_equil, LLT=LLT, d=d, e=e,
						z=z, z12=z12, zt=zt, zt12=zt12,
						zprev=zprev, rho=rho[0])
				elif save_equil:
					savez(filename, 
						A_equil=A_equil, d=d, e=e,
						z=z, z12=z12, zt=zt, zt12=zt12,
						zprev=zprev, rho=rho[0])
				else:
					savez(filename, 
						z=z, z12=z12, zt=zt, zt12=zt12,
						zprev=zprev, rho=rho[0])

			def __del__(self):
				if self.c_solver is not None:
					pogslib.pogs_finish(self.c_solver)

		self.Solver = Solver







