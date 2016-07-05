from numpy import zeros, ones, ndarray, savez, load as np_load
from numpy.linalg import cholesky
from ctypes import c_void_p
from os import path
from optkit.types.pogs.common import PogsCommonTypes

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
				if not isinstance(A, ndarray) or len(A.shape) != 2:
					raise TypeError('input must be a 2-d {}'.format(ndarray))

				self.__backend = backend
				self.shape = (self.m, self.n) = (m, n) = A.shape
				self.A = A.astype(lib.pyfloat)
				self.A_ptr = A_ptr = self.A.ctypes.data_as(lib.ok_float_p)
				self.__f = zeros(m).astype(lib.function)
				self.__f_ptr = self.__f.ctypes.data_as(lib.function_p)
				self.__f_c = lib.function_vector(self.m, self.__f_ptr)
				self.__g = zeros(n).astype(lib.function)
				self.__g_ptr = self.__g.ctypes.data_as(lib.function_p)
				self.__g_c = lib.function_vector(self.n, self.__g_ptr)
				self.layout = layout = lib.enums.CblasRowMajor if \
					A.flags.c_contiguous else lib.enums.CblasColMajor
				self.__c_solver = None

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
				if self.c_solver is None:
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

			def load(self, directory, name, allow_cholesky=True):
				filename = path.join(directory, name)
				if not '.npz' in name:
					filename += '.npz'

				err = 0
				try:
					data = np_load(filename)
				except:
					data = {}


				if 'A_equil' in data:
					A_equil = data['A_equil'].astype(lib.pyfloat)
				elif path.exists(path.join(directory, 'A_equil.npy')):
					A_equil = np_load(path.join(directory,
						'A_equil.npy')).astype(lib.pyfloat)
				else:
					err = 1

				if not err and 'd' in data:
					d = data['d'].astype(lib.pyfloat)
				elif not err and path.exists(path.join(directory, 'd.npy')):
					d = np_load(path.join(directory, 'd.npy')).astype(
								lib.pyfloat)
				else:
					err = 1

				if not err and 'e' in data:
					e = data['e'].astype(lib.pyfloat)
				elif not err and path.exists(path.join(directory, 'e.npy')):
					e = np_load(path.join(directory, 'e.npy')).astype(
								lib.pyfloat)
				else:
					err = 1

				if not err and 'LLT' in data:
					LLT = data['LLT'].astype(lib.pyfloat)
				elif not err and path.exists(path.join(directory, 'LLT.npy')):
					LLT = np_load(path.join(directory, 'LLT.npy')).astype(
								lib.pyfloat)
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
					LLT = cholesky(AA)

				else:
					err = 1

				if err:
					snippet = '`LLT`, ' if lib.direct else ''
					raise ValueError('Minimal requirements to load solver '
							   'not met. Specified file must contain '
							   'at least one .npz file with entries "A_equil",'
							   ' "LLT", "d", and "e", or the specified folder '
							   'must contain .npy files of the same names.')

				if 'z' in data:
					z = data['z'].astype(lib.pyfloat)
				else:
					z = zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'z12' in data:
					z12 = data['z12'].astype(lib.pyfloat)
				else:
					z12 = zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'zt' in data:
					zt = data['zt'].astype(lib.pyfloat)
				else:
					zt = zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'zt12' in data:
					zt12 = data['zt12'].astype(lib.pyfloat)
				else:
					zt12 = zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'zprev' in data:
					zprev = data['zprev'].astype(lib.pyfloat)
				else:
					zprev = zeros(self.m + self.n, dtype=lib.pyfloat)

				if 'rho' in data:
					rho = lib.pyfloat(data['rho'])
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
				A_equil = zeros((self.m, self.n), dtype=lib.pyfloat)

				LLT = zeros((mindim, mindim), dtype=lib.pyfloat)
				LLT_ptr = LLT.ctypes.data_as(lib.ok_float_p)

				if A_equil.flags.c_contiguous:
					order = lib.enums.CblasRowMajor
				else:
					order = lib.enums.CblasColMajor

				d = zeros(self.m, dtype=lib.pyfloat)
				e = zeros(self.n, dtype=lib.pyfloat)
				z = zeros(self.m + self.n, dtype=lib.pyfloat)
				z12 = zeros(self.m + self.n, dtype=lib.pyfloat)
				zt = zeros(self.m + self.n, dtype=lib.pyfloat)
				zt12 = zeros(self.m + self.n, dtype=lib.pyfloat)
				zprev = zeros(self.m + self.n, dtype=lib.pyfloat)
				rho = zeros(1, dtype=lib.pyfloat)

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

				if isinstance(LLT, ndarray) and save_factorization:
					savez(filename, A_equil=A_equil, LLT=LLT, d=d, e=e, z=z,
						  z12=z12, zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
				elif save_equil:
					savez(filename, A_equil=A_equil, d=d, e=e, z=z, z12=z12,
						  zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
				else:
					savez(filename, z=z, z12=z12, zt=zt, zt12=zt12,
						  zprev=zprev, rho=rho[0])

		self.Solver = Solver