from optkit.compat import *

import numpy as np

from optkit.utils.pyutils import const_iterator
from optkit.utils.proxutils import func_eval_python

class PogsCommonTypes(object):
	def __init__(self, lib):
		PogsSettings = lib.pogs_settings
		PogsInfo = lib.pogs_info
		PogsOutput = lib.pogs_output

		class Objective(object):
			def __init__(self, n, **params):
				self.enums = lib.function_enums
				self.size = n
				self.__h = np.zeros(self.size, dtype=int)
				self.__a = np.ones(self.size)
				self.__b = np.zeros(self.size)
				self.__c = np.ones(self.size)
				self.__d = np.zeros(self.size)
				self.__e = np.zeros(self.size)
				self.__s = np.ones(self.size)
				if 'f' in params:
					self.copy_from(params['f'])
				else:
					self.set(**params)

			def eval(self, vec):
				if self.size == 0:
					return 0.

				vec = np.reshape(np.array(vec), np.size(vec,))
				if vec.size != self.size:
					raise ValueError(
							'argument `vec` must have same length as '
							'this {} object.\nlength {}: {}\nlength '
							'vec: {}'
							''.format(
									Objective, Objective, self.size,
									vec.size))

				return func_eval_python(self.list(lib.function), vec)


			def copy_from(self, obj, start_index_target=0,
						  start_index_source=0, n_values=None):
				if self.size == 0:
					return
				if not isinstance(obj, Objective):
					raise TypeError(
							'argument `obj` must be of type {}'
							''.format(Objective))

				start_source = min(max(start_index_source, 0), obj.size)
				start_target = min(max(start_index_target, 0), self.size)

				end_source = obj.size
				if n_values is not None:
					end_source = min(start_source + int(n_values), end_source)
				end_target = start_target + (end_source - start_source)

				self.__h[start_target : end_target] = obj.__h[
						start_source : end_source]
				self.__a[start_target : end_target] = obj.__a[
						start_source : end_source]
				self.__b[start_target : end_target] = obj.__b[
						start_source : end_source]
				self.__c[start_target : end_target] = obj.__c[
						start_source : end_source]
				self.__d[start_target : end_target] = obj.__d[
						start_source : end_source]
				self.__e[start_target : end_target] = obj.__e[
						start_source : end_source]
				self.__s[start_target : end_target] = obj.__s[
						start_source : end_source]

			def list(self, function_t):
				return [
						function_t(
							self.__h[t], self.__a[t], self.__b[t],
							self.__c[t], self.__d[t], self.__e[t],
							self.__s[t],
							)
						for t in xrange(self.size)]

			@property
			def arrays(self):
				return self.h, self.a, self.b, self.c, self.d, self.e, self.s

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

			@property
			def s(self):
				return self.__s

			def set(self, **params):
				if self.size == 0:
					return

				start = int(params['start']) if 'start' in params else 0
				end = int(params['end']) if 'end' in params else self.size

				if start < 0 : start = self.size + start
				if end < 0 : end = self.size + end

				r = params.pop('range', xrange(start,end))
				range_length = len(r)

				if range_length == 0:
					raise ValueError(
							'index range [{}:{}] results in length-0 '
							'array when python array slicing applied '
							'to an {} of length {}.'
							''.format(start, end, Objective, self.size))

				for item in ['a', 'b', 'c', 'd', 'e', 's', 'h']:
					if item in params:
						if isinstance(params[item],(list, np.ndarray)):
							if len(params[item]) != range_length:
								raise ValueError(
										'keyword argument {} of type '
										'{} is incomptably sized with '
										'the requested {} slice [{}:{}]'
										''.format(
												item, type(params[item]),
												Objective, start, end))

				if 'h' in params:
					if isinstance(params['h'], (int, str)):
						hval = const_iterator(self.enums.validate(params['h']),
											  range_length)
					elif isinstance(params['h'], (list, np.ndarray)):
						hval = map(self.enums.validate, params['h'])
					else:
						raise TypeError(
								'if specified, argument `h` must be '
								'one of {}, {}, {} or {}'
								''.format(int, str, list, np.ndarray))

					for idx, val in enumerate(hval):
						self.__h[r[idx]] = val

				if 'a' in params:
					if isinstance(params['a'], (int, float)):
						aval = const_iterator(params['a'], range_length)
					elif isinstance(params['a'], (list, np.ndarray)):
						aval = params['a']
					else:
						raise TypeError(
								'if specified, argument `a` must be '
								'one of {}, {}, {} or {}'
								''.format(int, float, list, np.ndarray))

					for idx, val in enumerate(aval):
						self.__a[r[idx]] = val

				if 'b' in params:
					if isinstance(params['b'], (int, float)):
						bval = const_iterator(params['b'], range_length)
					elif isinstance(params['b'], (list, np.ndarray)):
						bval = params['b']
					else:
						raise TypeError(
								'if specified, argument `b` must be '
								'one of {}, {}, {} or {}'
								''.format(int, float, list, np.ndarray))

					for idx, val in enumerate(bval):
						self.__b[r[idx]] = val

				if 'c' in params:
					if isinstance(params['c'], (int, float)):
						cval = const_iterator(params['c'], range_length)
					elif isinstance(params['c'], (list, np.ndarray)):
						cval = map(self.enums.validate_c, params['c'])
					else:
						raise TypeError(
								'if specified, argument `c` must be '
								'one of {}, {}, {} or {}'
								''.format(int, float, list, np.ndarray))

					for idx, val in enumerate(cval):
						self.__c[r[idx]] = val

				if 'd' in params:
					if isinstance(params['d'], (int, float)):
						dval = const_iterator(params['d'], range_length)
					elif isinstance(params['d'], (list, np.ndarray)):
						dval = params['d']
					else:
						raise TypeError(
								'if specified, argument `d` must be '
								'one of {}, {}, {} or {}'
								''.format(int, float, list, np.ndarray))

					for idx, val in enumerate(dval):
						self.__d[r[idx]] = val

				if 'e' in params:
					if isinstance(params['e'], (int, float)):
						e_val = const_iterator(params['e'], range_length)
					elif isinstance(params['e'], (list, np.ndarray)):
						e_val = map(self.enums.validate_e, params['e'])
					else:
						raise TypeError(
								'if specified, argument `e` must be '
								'one of {}, {}, {} or {}'
								''.format(int, float, list, np.ndarray))

					for idx, val in enumerate(e_val):
						self.__e[r[idx]] = val

				if 's' in params:
					if isinstance(params['s'], (int, float)):
						s_val = const_iterator(params['s'], range_length)
					elif isinstance(params['s'], (list, np.ndarray)):
						s_val = map(self.enums.validate_s, params['s'])
					else:
						raise TypeError(
								'if specified, argument `s` must be '
								'one of {}, {}, {} or {}'
								''.format(int, float, list, np.ndarray))

					for idx, val in enumerate(s_val):
						self.__s[r[idx]] = val

			def __str__(self):
				return str(
						'size: {}\nh: {}\na: {}\nb: {}\nc: {}\nd: '
						'{}\ne: {}\ns: {}'
						''.format(
								self.size, self.h, self.a, self.b, self.c,
								self.d, self.e, self.s))

		self.Objective = Objective

		class SolverSettings(object):
			def __init__(self, **options):
				self.c = PogsSettings(
						0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None)
				lib.set_default_settings(self.c)
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
				elif 'maxiters' in options:
					self.maxiter = options['maxiters']
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
					self.x0 = options['x0'].ctypes.data_as(lib.ok_float_p)
				if 'nu0' in options:
					self.nu0 = options['nu0'].ctypes.data_as(ib.ok_float_p)

			@property
			def alpha(self):
				return self.c.alpha

			@alpha.setter
			def alpha(self, alpha):
				if not isinstance(alpha, (float, int)):
					raise TypeError(
							'argument `alpha` must be {} or {}'
							''.format(float, int))
				elif alpha < 0:
					raise ValueError('argument `alpha` must be >= 0')
				else:
					self.c.alpha = alpha

			@property
			def rho(self):
				return self.c.rho

			@rho.setter
			def rho(self, rho):
				if not isinstance(rho, (float, int)):
					raise TypeError(
							'argument `rho` must be {} or {}'
							''.format(float, int))
				elif rho < 0:
					raise ValueError('argument `rho` must be >= 0')
				else:
					self.c.rho = rho

			@property
			def abstol(self):
				return self.c.abstol

			@abstol.setter
			def abstol(self, abstol):
				if not isinstance(abstol, (float, int)):
					raise TypeError(
							'argument `abstol` must be {} or {}'
							''.format(float, int))
				elif abstol < 0:
					raise ValueError('argument `abstol` must be >= 0')
				else:
					self.c.abstol = abstol

			@property
			def reltol(self):
				return self.c.reltol

			@reltol.setter
			def reltol(self, reltol):
				if not isinstance(reltol, (float, int)):
					raise TypeError(
							'argument `reltol` must be {} or {}'
							''.format(float, int))
				elif reltol < 0:
					raise ValueError('argument `reltol` must be >= 0')
				else:
					self.c.reltol = reltol

			@property
			def maxiter(self):
				return self.c.maxiter

			@maxiter.setter
			def maxiter(self, maxiter):
				if not isinstance(maxiter, int):
					raise TypeError(
							'argument `maxiter` must be of type {}'
							''.format(int))
				elif maxiter < 0:
					raise ValueError('argument `maxiter` must be >= 0')
				else:
					self.c.maxiter = maxiter

			@property
			def verbose(self):
				return self.c.verbose

			@verbose.setter
			def verbose(self, verbose):
				if not isinstance(verbose, int):
					raise TypeError(
							'argument `verbose` must be of type {}'
							''.format(int))
				elif verbose < 0:
					raise ValueError('argument `verbose` must be >= 0')
				else:
					self.c.verbose = verbose

			@property
			def suppress(self):
				return self.c.suppress

			@suppress.setter
			def suppress(self, suppress):
				if not isinstance(suppress, int):
					raise TypeError(
							'argument `suppress` must be of type {}'
							''.format(int))
				elif suppress < 0:
					raise ValueError(
							'argument `suppress` must be >= 0')
				else:
					self.c.suppress = suppress

			@property
			def adaptiverho(self):
				return self.c.adaptiverho

			@adaptiverho.setter
			def adaptiverho(self, adaptiverho):
				if not isinstance(adaptiverho, (int, bool)):
					raise TypeError(
							'argument `adaptiverho` must be of type {} '
							'or {}'.format(int, bool))
				elif adaptiverho not in (0, 1, True, False):
					raise ValueError(
							'argument `adaptiverho` must be 0 or 1')
				else:
					self.c.adaptiverho = int(adaptiverho)

			@property
			def gapstop(self):
				return self.c.gapstop

			@gapstop.setter
			def gapstop(self, gapstop):
				if not isinstance(gapstop, (int, bool)):
					raise TypeError(
							'argument `gapstop` must be of type {} or '
							'{}'.format(int, bool))
				elif gapstop not in (0, 1, True, False):
					raise ValueError(
							'argument `gapstop` must be 0 or 1')
				else:
					self.c.gapstop = int(gapstop)

			@property
			def resume(self):
				return self.c.resume

			@resume.setter
			def resume(self, resume):
				if not isinstance(resume, (int, bool)):
					raise TypeError(
							'argument `resume` must be of type {} or {}'
							''.format(int, bool))
				elif resume not in (0, 1, True, False):
					raise ValueError('argument `resume` must be 0 or 1')
				else:
					self.c.resume = int(resume)

			@property
			def x0(self):
				return self.c.x0

			@x0.setter
			def x0(self, x0):
				if not isinstance(x0, np.ndarray):
					raise TypeError(
							'argument `x0` must be of type {}'
							''.format(np.ndarray))
				else:
					self._x0py = x0.astype(lib.pyfloat)
					self.c.x0 = self._x0py.ctypes.data_as(lib.ok_float_p)

			@property
			def nu0(self):
				return self.c.nu0

			@nu0.setter
			def nu0(self, nu0):
				if not isinstance(x0, np.ndarray):
					raise TypeError(
							'argument `nu0` must be of type {}'
							''.format(np.ndarray))
				else:
					self._n0py = nu0.astype(lib.pyfloat)
					self.c.nu0 = self._n0py.ctypes.data_as(lib.ok_float_p)

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
				self.c = PogsInfo()

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
						'iterations: {}\n'.format(self.iters)).join(
						'objective: {}\n'.format(self.objval)).join(
						'rho: {}\n'.format(self.rho)).join(
						'setup time: {}\n'.format(self.setup_time)).join(
						'solve time: {}\n'.format(self.solve_time)))

		self.SolverInfo = SolverInfo

		class SolverOutput(object):
			def __init__(self, m, n):
				self.x = np.zeros(n).astype(lib.pyfloat)
				self.y = np.zeros(m).astype(lib.pyfloat)
				self.mu = np.zeros(n).astype(lib.pyfloat)
				self.nu = np.zeros(m).astype(lib.pyfloat)
				self.c = PogsOutput(
						self.x.ctypes.data_as(lib.ok_float_p),
						self.y.ctypes.data_as(lib.ok_float_p),
						self.mu.ctypes.data_as(lib.ok_float_p),
						self.nu.ctypes.data_as(lib.ok_float_p))

			def __str__(self):
				return str(
						'x:\n{}\ny:\n{}\nmu:\n{}\nnu:\n{}\n'.format(
						str(self.x), str(self.y),
						str(self.mu), str(self.nu)))

		self.SolverOutput = SolverOutput