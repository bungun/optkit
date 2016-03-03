from numpy import zeros, ones, ndarray, savez, load as np_load
from ctypes import c_void_p
from os import path

def const_iterator(value, iters):
	for i in xrange(iters):
		yield value

class PogsTypes(object):
	def __init__(self, backend):
		PogsSettings = backend.pogs.pogs_settings
		PogsInfo = backend.pogs.pogs_info
		PogsOutput = backend.pogs.pogs_output
		pogslib = backend.pogs
		denselib = backend.dense
		proxlib = backend.prox

		# TODO: set direct/indirect, e.g.,
		# pogslib_direct = backend.pogs_direct
		# pogslib_indirect = backend.pogs_indirect


		class Objective(object):
			def __init__(self, n, **params):
				self.enums = proxlib.enums
				self.size = n
				self.__h = zeros(self.size, dtype=int)
				self.__a = ones(self.size)
				self.__b = zeros(self.size)
				self.__c = ones(self.size)
				self.__d = zeros(self.size)
				self.__e = zeros(self.size)
				if 'f' in params:
					self.copy_from(params['f'])
				else:
					self.set(**params)


			def copy_from(self, obj):
				if not isinstance(obj, Objective):
					raise TypeError('argument "obj" must be of type {}'.format(
									 Objective))
				if not obj.size == self.size:
					raise ValueError("Incompatible dimensions")
				self.__h[:] = obj.__h[:]

			def list(self, function_t):
				return [function_t(*t) for t in self.terms]

			@property
			def arrays(self):
				return self.__h, self.__a, self.__b, self.__c, self.__d, \
					   self.__e

			@property
			def h(self):
				return self.__h

			@property
			def a(self):
				return self.__a

			@property
			def b(self):
				return self.__b

			@property
			def c(self):
				return self.__c

			@property
			def d(self):
				return self.__d

			@property
			def e(self):
				return self.__e


			def set(self, **params):
				start = int(params['start']) if 'start' in params else 0
				end = int(params['end']) if 'end' in params else self.size

				if start < 0 : start = self.size + start
				if end < 0 : end = self.size + end


				r = params.pop('range', xrange(start,end))
				range_length = len(r)

				if  range_length == 0:
					ValueError('index range [{}:{}] results in length-0 array '
							   'when python array slicing applied to an '
							   '{} of length {}.'.format(start, end, Objective,
							   	self.size))

				for item in ['a', 'b', 'c', 'd', 'e', 'h']:
					if item in params:
						if isinstance(params[item],(list, ndarray)):
							if len(params[item]) != range_length:
								ValueError('keyword argument {} of type {} '
										   'is incomptably sized with the '
										   'requested {} slice [{}:{}]'.format(
										   	item, type(params(item)),
										   	Objective, start, end))


				if 'h' in params:
					if isinstance(params['h'], (int, str)):
						hval = const_iterator(self.enums.validate(params['h']),
											  range_length)
					elif isinstance(params['h'], (list, ndarray)):
						hval = map(self.enums.validate, params['h'])
					else:
						raise TypeError('if specified, argument "h" must be '
										'one of {}, {}, {} or {}'.format(
										int, str, list, ndarray))

					for idx, val in enumerate(hval):
						self.__h[r[idx]] = val

				if 'a' in params:
					if isinstance(params['a'], (int, float)):
						aval = const_iterator(params['a'], range_length)
					elif isinstance(params['a'], (list, ndarray)):
						aval = params['a']
					else:
						raise TypeError('if specified, argument "a" must be '
										'one of {}, {}, {} or {}'.format(
										int, float, list, ndarray))

					for idx, val in enumerate(aval):
						self.__a[r[idx]] = val

				if 'b' in params:
					if isinstance(params['b'], (int, float)):
						bval = const_iterator(params['b'], range_length)
					elif isinstance(params['b'], (list, ndarray)):
						bval = params['b']
					else:
						raise TypeError('if specified, argument "b" must be '
										'one of {}, {}, {} or {}'.format(
										int, float, list, ndarray))

					for idx, val in enumerate(bval):
						self.__b[r[idx]] = val

				if 'c' in params:
					if isinstance(params['c'], (int, float)):
						cval = const_iterator(
								self.enums.validate_ce(params['c']),
								range_length)
					elif isinstance(params['c'], (list, ndarray)):
						cval = params['c']
					else:
						raise TypeError('if specified, argument "c" must be '
										'one of {}, {}, {} or {}'.format(
										int, float, list, ndarray))

					for idx, val in enumerate(cval):
						self.__c[r[idx]] = val

				if 'd' in params:
					if isinstance(params['d'], (int, float)):
						dval = const_iterator(params['d'], range_length)
					elif isinstance(params['d'], (list, ndarray)):
						dval = params['d']
					else:
						raise TypeError('if specified, argument "d" must be '
										'one of {}, {}, {} or {}'.format(
										int, float, list, ndarray))

					for idx, val in enumerate(dval):
						self.__d[r[idx]] = val

				if 'e' in params:
					if isinstance(params['e'], (int, float)):
						e_val = const_iterator(
								self.enums.validate_ce(params['e']),
								range_length)
					elif isinstance(params['e'], (list, ndarray)):
						e_val = params['e']
					else:
						raise TypeError('if specified, argument "e" must be '
										'one of {}, {}, {} or {}'.format(
										int, float, list, ndarray))

					for idx, val in enumerate(e_val):
						self.__e[r[idx]] = val

			def __str__(self):
				return str("size:\nh: {}\na: {}\nb: {}\n"
					"c: {}\nd: {}\ne: {}".format(self.size,
					self.h, self.a, self.b, self.c, self.d, self.e))


		self.Objective = Objective

		class SolverSettings(object):
			def __init__(self, **options):
				self.c = PogsSettings(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None,
									  None)
				pogslib.set_default_settings(self.c)
				self.update(**options)

			def update(self, **options):
				if 'alpha' in options:
					self.alpha = options['alpha']
				if 'rho' in options:
					self.rho = options['rho']
				if 'abstol' in options:
					self.abstol = options['abstol']
				if 'reltol' in options:
					self.reltol = options['reltol']
				if 'maxiter' in options:
					self.maxiter = options['maxiter']
				if 'verbose' in options:
					self.verbose = options['verbose']
				if 'suppress' in options:
					self.suppress = options['suppress']
				if 'adaptiverho' in options:
					self.adaptiverho = options['adaptiverho']
				if 'gapstop' in options:
					self.gapstop = options['gapstop']
				if 'resume' in options:
					self.resume = options['resume']
				if 'x0' in options:
					self.x0 = options['x0'].ctypes.data_as(
														denselib.ok_float_p)
				if 'nu0' in options:
					self.nu0 = options['nu0'].ctypes.data_as(
														denselib.ok_float_p)

			@property
			def alpha(self):
				return self.c.alpha

			@alpha.setter
			def alpha(self, alpha):
				if not isinstance(alpha, (float, int)):
					raise TypeError('argument "alpha" must be {} or {}'.format(
									float, int))
				elif alpha < 0:
					raise ValueError('argument "alpha" must be >= 0')
				else:
					self.c.alpha = alpha

			@property
			def rho(self):
				return self.c.rho

			@rho.setter
			def rho(self, rho):
				if not isinstance(rho, (float, int)):
					raise TypeError('argument "rho" must be {} or {}'.format(
									float, int))
				elif rho < 0:
					raise ValueError('argument "rho" must be >= 0')
				else:
					self.c.rho = rho

			@property
			def abstol(self):
				return self.c.abstol

			@abstol.setter
			def abstol(self, abstol):
				if not isinstance(abstol, (float, int)):
					raise TypeError('argument "abstol" must be {} or '
									'{}'.format(float, int))
				elif abstol < 0:
					raise ValueError('argument "abstol" must be >= 0')
				else:
					self.c.abstol = abstol

			@property
			def reltol(self):
				return self.c.reltol

			@reltol.setter
			def reltol(self, reltol):
				if not isinstance(reltol, (float, int)):
					raise TypeError('argument "reltol" must be {} or '
									'{}'.format(float, int))
				elif reltol < 0:
					raise ValueError('argument "reltol" must be >= 0')
				else:
					self.c.reltol = reltol

			@property
			def maxiter(self):
				return self.c.maxiter

			@maxiter.setter
			def maxiter(self, maxiter):
				if not isinstance(maxiter, int):
					raise TypeError('argument "maxiter" must be of '
									'type {}'.format(int))
				elif maxiter < 0:
					raise ValueError('argument "maxiter" must be >= 0')
				else:
					self.c.maxiter = maxiter

			@property
			def verbose(self):
				return self.c.verbose

			@verbose.setter
			def verbose(self, verbose):
				if not isinstance(verbose, int):
					raise TypeError('argument "verbose" must be of '
									'type {}'.format(int))
				elif verbose < 0:
					raise ValueError('argument "verbose" must be >= 0')
				else:
					self.c.verbose = verbose

			@property
			def suppress(self):
				return self.c.suppress

			@suppress.setter
			def suppress(self, suppress):
				if not isinstance(suppress, int):
					raise TypeError('argument "suppress" must be of '
									'type {}'.format(int))
				elif suppress < 0:
					raise ValueError('argument "suppress" must be >= 0')
				else:
					self.c.suppress = suppress

			@property
			def adaptiverho(self):
				return self.c.adaptiverho

			@adaptiverho.setter
			def adaptiverho(self, adaptiverho):
				if not isinstance(adaptiverho, (int, bool)):
					raise TypeError('argument "adaptiverho" must be of '
									'type {} or {}'.format(int, bool))
				elif adaptiverho not in (0, 1, True, False):
					raise ValueError('argument "adaptiverho" must be 0 or 1')
				else:
					self.c.adaptiverho = int(adaptiverho)

			@property
			def gapstop(self):
				return self.c.gapstop

			@gapstop.setter
			def gapstop(self, gapstop):
				if not isinstance(gapstop, (int, bool)):
					raise TypeError('argument "gapstop" must be of '
									'type {} or {}'.format(int, bool))
				elif gapstop not in (0, 1, True, False):
					raise ValueError('argument "gapstop" must be 0 or 1')
				else:
					self.c.gapstop = int(gapstop)

			@property
			def resume(self):
				return self.c.resume

			@resume.setter
			def resume(self, resume):
				if not isinstance(resume, (int, bool)):
					raise TypeError('argument "resume" must be of '
									'type {} or {}'.format(int, bool))
				elif resume not in (0, 1, True, False):
					raise ValueError('argument "resume" must be 0 or 1')
				else:
					self.c.resume = int(resume)

			@property
			def x0(self):
				return self.c.x0

			@x0.setter
			def x0(self, x0):
				if not isinstance(x0, ndarray):
					raise TypeError('argument "x0" must be of '
									'type {}'.format(ndarray))
				else:
					self._x0py = x0.astype(denselib.pyfloat)
					self.c.x0 = self._x0py.ctypes.data_as(denselib.ok_float_p)

			@property
			def nu0(self):
				return self.c.nu0

			@nu0.setter
			def nu0(self, nu0):
				if not isinstance(x0, ndarray):
					raise TypeError('argument "nu0" must be of '
									'type {}'.format(ndarray))
				else:
					self._n0py = nu0.astype(denselib.pyfloat)
					self.c.nu0 = self._n0py.ctypes.data_as(denselib.ok_float_p)

			def __str__(self):
				return str(
						'alpha: {}\n'.format(self.alpha).join(
						'rho: {}\n'.format(self.rho)),join(
						'abstol: {}\n'.format(self.absoltol)).join(
						'reltol: {}\n'.format(self.reltol)).join(
						'maxiter: {}\n'.format(self.maxiter)).join(
						'verbose: {}\n'.format(self.verbose)).join(
						'suppress: {}\n'.format(self.suppress)).join(
						'adaptiverho: {}\ngapstop: {}\nwarmstart: {}\n'
						'resume: {}\n'.format(self.resume)).join(
						'x0: {}\n'.format(self.x0)).join(
						'nu0: {}\n'.format(self.c.nu0)))

		self.SolverSettings = SolverSettings

		class SolverInfo(object):
			def __init__(self):
				self.c = PogsInfo(0, 0, 0, 0, 0, 0, 0)

			@property
			def err(self):
				return self.c.err

			@property
			def iters(self):
				return self.c.k

			@property
			def solve_time(self):
				return self.c.solve_time

			@property
			def setup_time(self):
				return self.c.setup_time

			@property
			def error(self):
			    return self.c.error

			@property
			def converged(self):
			    return self.c.converged

			@property
			def objval(self):
			    return self.c.obj

			@property
			def rho(self):
				return self.c.rho

			def __str__(self):
				return str(
						'error: {}\n'.format(self.err).join(
						'converged: {}\n'.format(self.converged)).join(
						'iterations: {}\n'.format(self.k)).join(
						'objective: {}\n'.format(self.objval)).join(
						'rho: {}\n'.format(self.rho)).join(
						'setup time: {}\n'.format(self.setup_time)).join(
						'solve time: {}\n'.format(self.solve_time)))

		self.SolverInfo = SolverInfo

		class SolverOutput(object):
			def __init__(self, m, n):
				self.x = zeros(n, dtype=denselib.pyfloat)
				self.y = zeros(m, dtype=denselib.pyfloat)
				self.mu = zeros(n, dtype=denselib.pyfloat)
				self.nu = zeros(m, dtype=denselib.pyfloat)
				self.c = PogsOutput(
						self.x.ctypes.data_as(denselib.ok_float_p),
						self.y.ctypes.data_as(denselib.ok_float_p),
						self.mu.ctypes.data_as(denselib.ok_float_p),
						self.nu.ctypes.data_as(denselib.ok_float_p))

			def __str__(self):
				return str(
						'x:\n{}\ny:\n{}\nmu:\n{}\nnu:\n{}\n'.format(
						str(self.x), str(self.y),
						str(self.mu), str(self.nu)))

		self.SolverOutput = SolverOutput

		class Solver(object):
			def __init__(self, A, *args, **options):
				try:
					assert isinstance(A, ndarray)
					assert len(A.shape) == 2
				except:
					raise TypeError('input must be a 2-d {}'.format(ndarray))

				self.shape = (self.m, self.n) = (m, n) = A.shape
				self.A = A.astype(denselib.pyfloat)
				self.__f = zeros(n).astype(proxlib.function)
				self.__f_ptr = self.__f.ctypes.data_as(proxlib.function_p)
				self.__f_c = proxlib.function_vector(self.m, self.__f_ptr)
				self.__g = zeros(n).astype(proxlib.function)
				self.__g_ptr = self.__g.ctypes.data_as(proxlib.function_p)
				self.__g_c = proxlib.function_vector(self.n, self.__g_ptr)
				self.layout = layout = denselib.enums.CblasRowMajor if \
					A.flags.c_contiguous else denselib.enums.CblasColMajor

				if 'no_init' not in args:
					self.c_solver = pogslib.pogs_init(
							self.A.ctypes.data_as(denselib.ok_float_p), m , n,
							layout, pogslib.enums.EquilSinkhorn)
				else:
					self.c_solver = None

				self.settings = SolverSettings()
				self.info = SolverInfo()
				self.output = SolverOutput(m, n)
				self.settings.update(**options)
				self.first_run = True
				backend.increment_cobject_count()

			def __update_function_vectors(self, f, g):
				for i in xrange(f.size):
					self.__f[i] = proxlib.function(f.h[i], f.a[i], f.b[i],
												   f.c[i], f.d[i], f.e[i])

				for j in xrange(g.size):
					self.__g[j] = proxlib.function(g.h[j], g.a[j], g.b[j],
												   g.c[j], g.d[j], g.e[j])

			def solve(self, f, g, **options):
				if self.c_solver is None:
					raise ValueError(
							'No solver intialized, solve() call invalid')

				if not isinstance(f, Objective) and isinstance(g, Objective):
					raise TypeError(
						'inputs f, g must be of type {} \nprovided: {}, '
						'{}'.format(Objective, type(f), type(g)))

				if not (f.size == self.m and g.size == self.n):
					raise ValueError(
						'inputs f, g not compatibly sized with solver'
						'\nsolver dimensions ({}, {})\n provided: '
						'({}{})'.format(self.m, self.n, f.size, g.size))


				# TODO : logic around resume, warmstart, rho input

				self.__update_function_vectors(f, g)
				self.settings.update(**options)
				pogslib.pogs_solve(self.c_solver, self.__f_c, self.__g_c,
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
					A_equil = data['A_equil'].astype(denselib.pyfloat)
				elif path.exists(path.join(directory, 'A_equil.npy')):
					A_equil = np_load(path.join(directory,
						'A_equil.npy')).astype(denselib.pyfloat)
				else:
					err = 1


				if not err and 'LLT' in data:
					LLT = data['LLT'].astype(denselib.pyfloat)
				elif path.exists(path.join(directory, 'LLT.npy')):
					LLT = np_load(path.join(directory, 'LLT.npy')).astype(
								denselib.pyfloat)
					LLT_ptr = LLT.ctypes.data_as(denselib.ok_float_p)
				else:
					if pogslib.direct:
						err = 1
					else:
						LLT_ptr = c_void_p()

				if not err:
					LLT_ptr = LLT.ctypes.data_as(denselib.ok_float_p)
				else:
					LLT_ptr = None


				if not err and 'd' in data:
					d = data['d'].astype(denselib.pyfloat)
				elif path.exists(path.join(directory, 'd.npy')):
					d = np_load(path.join(directory, 'd.npy')).astype(
								denselib.pyfloat)
				else:
					err = 1

				if not err and 'e' in data:
					e = data['e'].astype(denselib.pyfloat)
				elif path.exists(path.join(directory, 'e.npy')):
					e = np_load(path.join(directory, 'e.npy')).astype(
								denselib.pyfloat)
				else:
					err = 1

				if err:
					snippet = '`LLT`, ' if pogslib.direct else ''
					ValueError('Minimal requirements to load solver '
							   'not met. Specified file must contain '
							   'at least one .npz file with entries `A_equil`,'
							   ' {}`d`, and `e`, or the specified folder must'
							   ' contain .npy files of the same names.'.format(
							   snippet))

				if 'z' in data:
					z = data['z'].astype(denselib.pyfloat)
				else:
					z = zeros(self.m + self.n, dtype=denselib.pyfloat)

				if 'z12' in data:
					z12 = data['z12'].astype(denselib.pyfloat)
				else:
					z12 = zeros(self.m + self.n, dtype=denselib.pyfloat)

				if 'zt' in data:
					zt = data['zt'].astype(denselib.pyfloat)
				else:
					zt = zeros(self.m + self.n, dtype=denselib.pyfloat)

				if 'zt12' in data:
					zt12 = data['zt12'].astype(denselib.pyfloat)
				else:
					zt12 = zeros(self.m + self.n, dtype=denselib.pyfloat)

				if 'zprev' in data:
					zprev = data['zprev'].astype(denselib.pyfloat)
				else:
					zprev = zeros(self.m + self.n, dtype=denselib.pyfloat)

				if 'rho' in data:
					rho = denselib.pyfloat(data['rho'])
				else:
					rho = 1.

				order = denselib.enums.CblasRowMajor if \
					A_equil.flags.c_contiguous else denselib.enums.CblasColMajor

				if self.c_solver is not None:
					pogslib.pogs_finish(self.c_solver)

				self.c_solver = pogslib.pogs_load_solver(
						A_equil.ctypes.data_as(denselib.ok_float_p), LLT_ptr,
						d.ctypes.data_as(denselib.ok_float_p),
						e.ctypes.data_as(denselib.ok_float_p),
						z.ctypes.data_as(denselib.ok_float_p),
						z12.ctypes.data_as(denselib.ok_float_p),
						zt.ctypes.data_as(denselib.ok_float_p),
						zt12.ctypes.data_as(denselib.ok_float_p),
						zprev.ctypes.data_as(denselib.ok_float_p),
						rho, self.m, self.n, order)

			def save(self, directory, name, save_equil=True,
					 save_factorization=True):

				if self.c_solver is None:
					raise ValueError(
							'No solver intialized, save() call invalid')

				filename = path.join(directory, name)
				if not name.endswith('.npz'):
					filename.join('.npz')

				if not path.exists(directory):
					raise ValueError('specified directory does not exist')

				if path.exists(filename):
					raise ValueError('specified filepath already exists '
									 'and would be overwritten, aborting.')

				mindim = min(self.m, self.n)
				A_equil = zeros((self.m, self.n), dtype=denselib.pyfloat)

				if pogslib.direct:
					LLT = zeros((mindim, mindim), dtype=denselib.pyfloat)
					LLT_ptr = LLT.ctypes.data_as(denselib.ok_float_p)
				else:
					LLT = c_void_p()
					LLT_ptr = LLT

				if A_equil.flags.c_contiguous:
					order = denselib.enums.CblasRowMajor
				else:
					order = denselib.enums.CblasColMajor

				d = zeros(self.m, dtype=denselib.pyfloat)
				e = zeros(self.n, dtype=denselib.pyfloat)
				z = zeros(self.m + self.n, dtype=denselib.pyfloat)
				z12 = zeros(self.m + self.n, dtype=denselib.pyfloat)
				zt = zeros(self.m + self.n, dtype=denselib.pyfloat)
				zt12 = zeros(self.m + self.n, dtype=denselib.pyfloat)
				zprev = zeros(self.m + self.n, dtype=denselib.pyfloat)
				rho = zeros(1, dtype=denselib.pyfloat)

				pogslib.pogs_extract_solver(
						self.c_solver,
						A_equil.ctypes.data_as(denselib.ok_float_p), LLT_ptr,
						d.ctypes.data_as(denselib.ok_float_p),
						e.ctypes.data_as(denselib.ok_float_p),
						z.ctypes.data_as(denselib.ok_float_p),
						z12.ctypes.data_as(denselib.ok_float_p),
						zt.ctypes.data_as(denselib.ok_float_p),
						zt12.ctypes.data_as(denselib.ok_float_p),
						zprev.ctypes.data_as(denselib.ok_float_p),
						rho.ctypes.data_as(denselib.ok_float_p), order)

				if isinstance(LLT, ndarray) and save_factorization:
					savez(filename, A_equil=A_equil, LLT=LLT, d=d, e=e, z=z,
						  z12=z12, zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
				elif save_equil:
					savez(filename, A_equil=A_equil, d=d, e=e, z=z, z12=z12,
						  zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
				else:
					savez(filename, z=z, z12=z12, zt=zt, zt12=zt12,
						  zprev=zprev, rho=rho[0])

			def __del__(self):
				backend.decrement_cobject_count()
				if self.c_solver is not None:
					pogslib.pogs_finish(self.c_solver,
										int(backend.device_reset_allowed))

		self.Solver = Solver