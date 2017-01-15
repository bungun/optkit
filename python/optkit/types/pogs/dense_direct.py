from optkit.compat import *

import os
import numpy as np
import numpy.linalg as la
from optkit.types.pogs.common import PogsCommonTypes

class DoubleCache(object):
	def __init__(self, npz_file=None, dictionary=None):
		self.__dictionary = {}
		self.__npz_cache = {}

		if isinstance(npz_file, DoubleCache):
			self.set_file(npz_file._DoubleCache__npz_cache)
			self.update(npz_file._DoubleCache__dictionary)
		else:
			self.set_file(npz_file)
		self.update(dictionary)

	def set_file(self, npz_file):
		if isinstance(npz_file, np.lib.npyio.NpzFile):
			self.__npz_cache = npz_file
		elif isinstance(npz_file, dict):
			self.update(npz_file)

	def update(self, dictionary):
		if isinstance(dictionary, dict):
			self.__dictionary.update(dictionary)

	def __contains__(self, key):
		return key in self.__npz_cache or key in self.__dictionary

	def __setitem__(self, key, item):
		self.__dictionary[key] = item

	def __getitem__(self, key):
		if key in self.__npz_cache:
			return self.__npz_cache[key]
		elif key in self.__dictionary:
			return self.__dictionary
		else:
			raise KeyError(
					'{} has no entry for key=`{}`'.format(DoubleCache, key))

class PogsDenseDirectTypes(PogsCommonTypes):
	def __init__(self, backend):
		PogsCommonTypes.__init__(self, backend)
		lib = backend.pogs
		PogsSettings = lib.pogs_settings
		PogsInfo = lib.pogs_info
		PogsOutput = lib.pogs_output
		Objective = self.Objective

		SolverSettings = self.SolverSettings
		SolverInfo = self.SolverInfo
		SolverOutput = self.SolverOutput

		class Solver(object):
			def __del__(self):
				self.__unregister_solver()

			def __init__(self, A, *args, **options):
				if not isinstance(A, np.ndarray) or len(A.shape) != 2:
					raise TypeError('input must be a 2-d {}'.format(np.ndarray))

				self.__backend = backend
				self.shape = (self.m, self.n) = (m, n) = A.shape
				self.A = A.astype(lib.pyfloat)
				self.A_ptr = A_ptr = self.A.ctypes.data_as(lib.ok_float_p)
				self.__f = np.zeros(m).astype(lib.function)
				self.__f_ptr = self.__f.ctypes.data_as(lib.function_p)
				self.__f_c = lib.function_vector(self.m, self.__f_ptr)
				self.__g = np.zeros(n).astype(lib.function)
				self.__g_ptr = self.__g.ctypes.data_as(lib.function_p)
				self.__g_c = lib.function_vector(self.n, self.__g_ptr)
				self.layout = layout = lib.enums.CblasRowMajor if \
					A.flags.c_contiguous else lib.enums.CblasColMajor
				self.__c_solver = None
				self.__cache = None
				self.__state = None

				if 'no_init' not in args:
					self.__register_solver(lib, lib.pogs_init(self.A_ptr, m, n,
							layout))

				self.settings = SolverSettings()
				self.info = SolverInfo()
				self.output = SolverOutput(m, n)
				self.settings.update(**options)
				self.first_run = True

			@property
			def c_solver(self):
			    return self.__c_solver

			def __register_solver(self, lib, solver):
				err = self.__backend.pogs.pogs_solver_exists(solver)
				if err > 0:
					raise RuntimeError('solver allocation failed')

				self.__backend.increment_cobject_count()
				self.__c_solver = solver

			def __unregister_solver(self):
				if self.__c_solver is None:
					return
				self.__backend.pogs.pogs_finish(self.c_solver, 0)
				self.__c_solver = None
				self.__backend.decrement_cobject_count()


			def __update_function_vectors(self, f, g):
				for i in xrange(f.size):
					self.__f[i] = lib.function(f.h[i], f.a[i], f.b[i],
												   f.c[i], f.d[i], f.e[i])

				for j in xrange(g.size):
					self.__g[j] = lib.function(g.h[j], g.a[j], g.b[j],
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
				lib.pogs_solve(self.c_solver, self.__f_c, self.__g_c,
								   self.settings.c, self.info.c, self.output.c)
				self.first_run = False

			def load_cache(self, cache, allow_cholesky=True, cache_extra=None):
				cache = DoubleCache(cache, cache_extra)
				err = 0

				if 'A_equil' in cache:
					A_equil = cache['A_equil'].astype(lib.pyfloat)
				else:
					err = 1

				if not err and 'd' in cache:
					d = cache['d'].astype(lib.pyfloat)
				else:
					err = 1

				if not err and 'e' in cache:
					e = cache['e'].astype(lib.pyfloat)
				else:
					err = 1

				if not err and 'LLT' in cache:
					LLT = cache['LLT'].astype(lib.pyfloat)
				elif not err and allow_cholesky:
					m, n = A_equil.shape
					mindim = min(m, n)
					if m >= n:
						AA = A_equil.T.dot(A_equil)
					else:
						AA = A_equil.dot(A_equil.T)

					mean_diag = 0
					for i in xrange(mindim):
						mean_diag += AA[i, i]

					mean_diag /= AA.shape[0]
					normA = sqrt()
					LLT = la.cholesky(AA)

				else:
					err = 1

				if err:
					snippet = '`LLT`, ' if lib.direct else ''
					raise ValueError('Minimal requirements to load solver '
							   'not met. Specified file(s) must contain '
							   'at least one .npz file with entries "A_equil",'
							   ' "LLT", "d", and "e", or the specified folder '
							   'must contain .npy files of the same names.')

				if 'z' in cache:
					z = cache['z'].astype(lib.pyfloat)
				else:
					z = np.zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'z12' in cache:
					z12 = cache['z12'].astype(lib.pyfloat)
				else:
					z12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'zt' in cache:
					zt = cache['zt'].astype(lib.pyfloat)
				else:
					zt = np.zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'zt12' in cache:
					zt12 = cache['zt12'].astype(lib.pyfloat)
				else:
					zt12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'zprev' in cache:
					zprev = cache['zprev'].astype(lib.pyfloat)
				else:
					zprev = np.zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'rho' in cache:
					rho = lib.pyfloat(cache['rho'])
				else:
					rho = 1.

				order = lib.enums.CblasRowMajor if \
					A_equil.flags.c_contiguous else lib.enums.CblasColMajor

				if self.c_solver is not None:
					self.__unregister_solver()

				self.__register_solver(lib, lib.pogs_load_solver(
						A_equil.ctypes.data_as(lib.ok_float_p),
						LLT.ctypes.data_as(lib.ok_float_p),
						d.ctypes.data_as(lib.ok_float_p),
						e.ctypes.data_as(lib.ok_float_p),
						z.ctypes.data_as(lib.ok_float_p),
						z12.ctypes.data_as(lib.ok_float_p),
						zt.ctypes.data_as(lib.ok_float_p),
						zt12.ctypes.data_as(lib.ok_float_p),
						zprev.ctypes.data_as(lib.ok_float_p),
						rho, self.m, self.n, order))

			def load(self, directory, name, allow_cholesky=True):
				filename = os.path.join(directory, name)
				if not '.npz' in name:
					filename += '.npz'

				try:
					data = DoubleCache(np.load(filename))
				except:
					data = DoubleCache()

				if not 'A_equil' in data:
					file = os.path.join(directory, 'A_equil.npy')
					if os.path.exists(file):
						data['A_equil'] = np.load(file).astype(lib.pyfloat)

				if not 'd' in data:
					file = os.path.join(directory, 'd.npy')
					if os.path.exists(file):
						data['d'] = np.load(file).astype(lib.pyfloat)

				if not 'e' in data:
					file = os.path.join(directory, 'e.npy')
					if os.path.exists(file):
						data['e'] = np.load(file).astype(lib.pyfloat)

				if not 'LLT' in data:
					file = os.path.join(directory, 'LLT.npy')
					if os.path.exists(file):
						data['LLT'] = np.load(file).astype(lib.pyfloat)

				if not 'z' in data:
					file = os.path.join(directory, 'z.npy')
					if os.path.exists(file):
						data['z'] = np.load(file).astype(lib.pyfloat)

				if not 'z12' in data:
					file = os.path.join(directory, 'z12.npy')
					if os.path.exists(file):
						data['z12'] = np.load(file).astype(lib.pyfloat)

				if not 'zt' in data:
					file = os.path.join(directory, 'zt.npy')
					if os.path.exists(file):
						data['zt'] = np.load(file).astype(lib.pyfloat)

				if not 'zt12' in data:
					file = os.path.join(directory, 'zt12.npy')
					if os.path.exists(file):
						data['zt12'] = np.load(file).astype(lib.pyfloat)

				if not 'zprev' in data:
					file = os.path.join(directory, 'zprev.npy')
					if os.path.exists(file):
						data['zprev'] = np.load(file).astype(lib.pyfloat)

				if 'rho' in data:
					rho = lib.pyfloat(data['rho'])
				else:
					data['rho'] = 1.0

				self.load_cache(data, allow_cholesky=allow_cholesky)

			# def load(self, directory, name, allow_cholesky=True):
			# 	filename = os.path.join(directory, name)
			# 	if not '.npz' in name:
			# 		filename += '.npz'

			# 	err = 0
			# 	try:
			# 		data = np.load(filename)
			# 	except:
			# 		data = {}


			# 	if 'A_equil' in data:
			# 		A_equil = data['A_equil'].astype(lib.pyfloat)
			# 	elif os.path.exists(os.path.join(directory, 'A_equil.npy')):
			# 		A_equil = np.load(os.path.join(directory,
			# 			'A_equil.npy')).astype(lib.pyfloat)
			# 	else:
			# 		err = 1

			# 	if not err and 'd' in data:
			# 		d = data['d'].astype(lib.pyfloat)
			# 	elif not err and os.path.exists(os.path.join(directory, 'd.npy')):
			# 		d = np.load(os.path.join(directory, 'd.npy')).astype(
			# 					lib.pyfloat)
			# 	else:
			# 		err = 1

			# 	if not err and 'e' in data:
			# 		e = data['e'].astype(lib.pyfloat)
			# 	elif not err and os.path.exists(os.path.join(directory, 'e.npy')):
			# 		e = np.load(os.path.join(directory, 'e.npy')).astype(
			# 					lib.pyfloat)
			# 	else:
			# 		err = 1

			# 	if not err and 'LLT' in data:
			# 		LLT = data['LLT'].astype(lib.pyfloat)
			# 	elif not err and os.path.exists(os.path.join(directory, 'LLT.npy')):
			# 		LLT = np.load(os.path.join(directory, 'LLT.npy')).astype(
			# 					lib.pyfloat)
			# 	elif not err and allow_cholesky:
			# 		m, n = A_equil.shape
			# 		mindim = min(m, n)
			# 		if m >= n:
			# 			AA = A_equil.T.dot(A_equil)
			# 		else:
			# 			AA = A_equil.dot(A_equil.T)


			# 		mean_diag = 0
			# 		for i in xrange(mindim):
			# 			mean_diag += AA[i, i]

			# 		mean_diag /= AA.shape[0]
			# 		normA = sqrt()
			# 		LLT = cholesky(AA)

			# 	else:
			# 		err = 1

			# 	if err:
			# 		snippet = '`LLT`, ' if lib.direct else ''
			# 		raise ValueError('Minimal requirements to load solver '
			# 				   'not met. Specified file must contain '
			# 				   'at least one .npz file with entries "A_equil",'
			# 				   ' "LLT", "d", and "e", or the specified folder '
			# 				   'must contain .npy files of the same names.')

			# 	if 'z' in data:
			# 		z = data['z'].astype(lib.pyfloat)
			# 	else:
			# 		z = np.zeros(self.m + self.n, dtype=lib.pyfloat)

			# 	if 'z12' in data:
			# 		z12 = data['z12'].astype(lib.pyfloat)
			# 	else:
			# 		z12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)

			# 	if 'zt' in data:
			# 		zt = data['zt'].astype(lib.pyfloat)
			# 	else:
			# 		zt = np.zeros(self.m + self.n, dtype=lib.pyfloat)

			# 	if 'zt12' in data:
			# 		zt12 = data['zt12'].astype(lib.pyfloat)
			# 	else:
			# 		zt12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)

			# 	if 'zprev' in data:
			# 		zprev = data['zprev'].astype(lib.pyfloat)
			# 	else:
			# 		zprev = np.zeros(self.m + self.n, dtype=lib.pyfloat)

			# 	if 'rho' in data:
			# 		rho = lib.pyfloat(data['rho'])
			# 	else:
			# 		rho = 1.

			# 	order = lib.enums.CblasRowMajor if \
			# 		A_equil.flags.c_contiguous else lib.enums.CblasColMajor

			# 	if self.c_solver is not None:
			# 		self.__unregister_solver()

			# 	self.__register_solver(lib, lib.pogs_load_solver(
			# 			A_equil.ctypes.data_as(lib.ok_float_p),
			# 			LLT.ctypes.data_as(lib.ok_float_p),
			# 			d.ctypes.data_as(lib.ok_float_p),
			# 			e.ctypes.data_as(lib.ok_float_p),
			# 			z.ctypes.data_as(lib.ok_float_p),
			# 			z12.ctypes.data_as(lib.ok_float_p),
			# 			zt.ctypes.data_as(lib.ok_float_p),
			# 			zt12.ctypes.data_as(lib.ok_float_p),
			# 			zprev.ctypes.data_as(lib.ok_float_p),
			# 			rho, self.m, self.n, order))

			@property
			def cache(self):
				if self.c_solver is None:
					return {}

				if self.__cache is not None:
					return {k: self.__cache[k] for k in self.__cache}

				self.__cache = {
						'A_equil': None,
						'd': None,
						'e': None,
						'LLT': None,
						'z': None,
						'z12': None,
						'zt': None,
						'zt12': None,
						'zprev': None,
						'rho': 1.0,
						'direct': lib.direct,
				}

				mindim = min(self.m, self.n)
				A_equil = np.zeros((self.m, self.n), dtype=lib.pyfloat)

				LLT = np.zeros((mindim, mindim), dtype=lib.pyfloat)
				LLT_ptr = LLT.ctypes.data_as(lib.ok_float_p)

				if A_equil.flags.c_contiguous:
					order = lib.enums.CblasRowMajor
				else:
					order = lib.enums.CblasColMajor

				d = np.zeros(self.m, dtype=lib.pyfloat)
				e = np.zeros(self.n, dtype=lib.pyfloat)
				z = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				z12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				zt = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				zt12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				zprev = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				rho = np.zeros(1, dtype=lib.pyfloat)

				# TODO: separate equil/factoriation copy from state variable copy
				lib.pogs_extract_solver(
						self.c_solver,
						A_equil.ctypes.data_as(lib.ok_float_p), LLT_ptr,
						d.ctypes.data_as(lib.ok_float_p),
						e.ctypes.data_as(lib.ok_float_p),
						z.ctypes.data_as(lib.ok_float_p),
						z12.ctypes.data_as(lib.ok_float_p),
						zt.ctypes.data_as(lib.ok_float_p),
						zt12.ctypes.data_as(lib.ok_float_p),
						zprev.ctypes.data_as(lib.ok_float_p),
						rho.ctypes.data_as(lib.ok_float_p), order)

				self.__cache['A_equil'] = A_equil
				self.__cache['d'] = d
				self.__cache['e'] = e
				self.__cache['LLT'] = LLT
				self.__cache['z'] = z
				self.__cache['z12'] = z12
				self.__cache['zt'] = zt
				self.__cache['zt12'] = zt12
				self.__cache['zprev'] = zprev
				self.__cache['rho'] = rho[0]

				return {k: self.__cache[k] for k in self.__cache}

			@property
			def state(self):
				return NotImplemented
				# TODO: this property can probably even go in PogsSolverCommon?

				# if self.c_solver is None:
					# return {}

				# z = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# z12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# zt = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# zt12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# zprev = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# rho = np.zeros(1, dtype=lib.pyfloat)

				# lib.pogs_extract_state(
				# 		self.c_solver,
				# 		z.ctypes.data_as(lib.ok_float_p),
				# 		z12.ctypes.data_as(lib.ok_float_p),
				# 		zt.ctypes.data_as(lib.ok_float_p),
				# 		zt12.ctypes.data_as(lib.ok_float_p),
				# 		zprev.ctypes.data_as(lib.ok_float_p),
				# 		rho.ctypes.data_as(lib.ok_float_p), order)

				# self.__state['z'] = z
				# self.__state['z12'] = z12
				# self.__state['zt'] = zt
				# self.__state['zt12'] = zt12
				# self.__state['zprev'] = zprev
				# self.__state['rho'] = rho[0]

				# return {k: self.__state[k] for k in self.__state}

			def save(self, directory, name, save_equil=True,
					 save_factorization=True):

				if self.c_solver is None:
					raise ValueError(
							'No solver intialized, save() call invalid')

				filename = os.path.join(directory, name)
				if not name.endswith('.npz'):
					filename += '.npz'

				if not os.path.exists(directory):
					raise ValueError('specified directory does not exist')

				if os.path.exists(filename):
					raise ValueError('specified filepath already exists '
									 'and would be overwritten, aborting.')

				cache = self.cache
				if not save_factorization:
					cache.pop('LLT', None)
				if not save_equil:
					cache.pop('A', None)
					cache.pop('d', None)
					cache.pop('e', None)
					cache.pop('LLT', None)

				np.savez(filename, **cache)
				return filename
				# mindim = min(self.m, self.n)
				# A_equil = np.zeros((self.m, self.n), dtype=lib.pyfloat)

				# LLT = np.zeros((mindim, mindim), dtype=lib.pyfloat)
				# LLT_ptr = LLT.ctypes.data_as(lib.ok_float_p)

				# if A_equil.flags.c_contiguous:
				# 	order = lib.enums.CblasRowMajor
				# else:
				# 	order = lib.enums.CblasColMajor

				# d = np.zeros(self.m, dtype=lib.pyfloat)
				# e = np.zeros(self.n, dtype=lib.pyfloat)
				# z = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# z12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# zt = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# zt12 = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# zprev = np.zeros(self.m + self.n, dtype=lib.pyfloat)
				# rho = np.zeros(1, dtype=lib.pyfloat)

				# lib.pogs_extract_solver(
				# 		self.c_solver,
				# 		A_equil.ctypes.data_as(lib.ok_float_p), LLT_ptr,
				# 		d.ctypes.data_as(lib.ok_float_p),
				# 		e.ctypes.data_as(lib.ok_float_p),
				# 		z.ctypes.data_as(lib.ok_float_p),
				# 		z12.ctypes.data_as(lib.ok_float_p),
				# 		zt.ctypes.data_as(lib.ok_float_p),
				# 		zt12.ctypes.data_as(lib.ok_float_p),
				# 		zprev.ctypes.data_as(lib.ok_float_p),
				# 		rho.ctypes.data_as(lib.ok_float_p), order)

				# if isinstance(LLT, np.ndarray) and save_factorization:
				# 	np.savez(filename, A_equil=A_equil, LLT=LLT, d=d, e=e, z=z,
				# 		  z12=z12, zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
				# elif save_equil:
				# 	np.savez(filename, A_equil=A_equil, d=d, e=e, z=z, z12=z12,
				# 		  zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
				# else:
				# 	np.savez(filename, z=z, z12=z12, zt=zt, zt12=zt12,
				# 		  zprev=zprev, rho=rho[0])

		self.Solver = Solver