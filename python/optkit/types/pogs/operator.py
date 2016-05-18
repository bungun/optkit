from numpy import zeros, ones, ndarray, savez, load as np_load
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from ctypes import c_void_p
from os import path
from optkit.types.operator import OperatorTypes
from optkit.types.pogs.common import PogsTypes

class PogsOperatorTypes(PogsTypes):
	def __init__(self, backend):
		PogsTypes.__init__(self, backend)
		PogsSettings = backend.pogs.pogs_settings
		PogsInfo = backend.pogs.pogs_info
		PogsOutput = backend.pogs.pogs_output
		lib = backend.pogs

		Operator = OperatorTypes(backend).AbstractLinearOperator

		class Solver(object):
			def __init__(self, A, **options):
				self.A = Operator(A)
				self.shape = (self.m, self.n) = (m, n) = self.A.shape
				self.__f = zeros(m).astype(lib.function)
				self.__f_ptr = self.__f.ctypes.data_as(lib.function_p)
				self.__f_c = lib.function_vector(self.m, self.__f_ptr)
				self.__g = zeros(n).astype(lib.function)
				self.__g_ptr = self.__g.ctypes.data_as(lib.function_p)
				self.__g_c = lib.function_vector(self.n, self.__g_ptr)
				self.layout = layout = lib.enums.CblasRowMajor if \
					A.flags.c_contiguous else lib.enums.CblasColMajor

				NO_INIT = bool(options.pop('no_init', False))
				DIRECT = int(options.pop('direct', False))
				EQUILNORM = float(options.pop('equil_norm', 1.))

				if not NO_INIT:
					self.__register_solver(lib, lib.pogs_init(
							self.A.c_ptr, DIRECT, EQUILNORM))
				else:
					self.c_solver = None

				self.settings = SolverSettings()
				self.info = SolverInfo()
				self.output = SolverOutput(m, n)
				self.settings.update(**options)
				self.first_run = True

			@property
			def c_solver(self):
			    return self.__c_solver

			def __register_solver(self, lib, solver):
				self.__c_solver = solver
				self.exit_call = lib.pogs_finish
				self.exit_arg = solver
				self.reset_on_exit = True

			def __unregister_solver(self):
				if self.c_solver is None:
					return
				self.exit_call(self.c_solver, 0)
				self.__c_solver = None
				self.exit_call = self._OptkitBaseTypeC__exit_default
				self.exit_arg = None
				self.reset_on_exit = False

			def __update_function_vectors(self, f, g):
				for i in xrange(f.size):
					self.__f[i] = lib.function(f.h[i], f.a[i], f.b[i],
												   f.c[i], f.d[i], f.e[i])

				for j in xrange(g.size):
					self.__g[j] = lib.function(g.h[j], g.a[j], g.b[j],
												   g.c[j], g.d[j], g.e[j])

# 			def solve(self, f, g, **options):
# 				if self.c_solver is None:
# 					raise ValueError(
# 							'No solver intialized, solve() call invalid')

# 				if not isinstance(f, Objective) and isinstance(g, Objective):
# 					raise TypeError(
# 						'inputs f, g must be of type {} \nprovided: {}, '
# 						'{}'.format(Objective, type(f), type(g)))

# 				if not (f.size == self.m and g.size == self.n):
# 					raise ValueError(
# 						'inputs f, g not compatibly sized with solver'
# 						'\nsolver dimensions ({}, {})\n provided: '
# 						'({}{})'.format(self.m, self.n, f.size, g.size))


# 				# TODO : logic around resume, warmstart, rho input

# 				self.__update_function_vectors(f, g)
# 				self.settings.update(**options)
# 			lib.pogs_solve(self.c_solver, self.__f_c, self.__g_c,
# 								   self.settings.c, self.info.c, self.output.c)
# 				self.first_run = False

# 			# def load(self, directory, name):
# 			# 	filename = path.join(directory, name)
# 			# 	if not '.npz' in name:
# 			# 		filename += '.npz'

# 			# 	err = 0
# 			# 	try:
# 			# 		data = np_load(filename)
# 			# 	except:
# 			# 		data = {}


# 			# 	if 'A_equil' in data:
# 			# 		A_equil = data['A_equil'].astype(lib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'A_equil.npy')):
# 			# 		A_equil = np_load(path.join(directory,
# 			# 			'A_equil.npy')).astype(lib.pyfloat)
# 			# 	else:
# 			# 		err = 1


# 			# 	if not err and 'LLT' in data:
# 			# 		LLT = data['LLT'].astype(lib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'LLT.npy')):
# 			# 		LLT = np_load(path.join(directory, 'LLT.npy')).astype(
# 			# 					lib.pyfloat)
# 			# 		LLT_ptr = LLT.ctypes.data_as(lib.ok_float_p)
# 			# 	else:
# 			# 		iflib.direct:
# 			# 			err = 1
# 			# 		else:
# 			# 			LLT_ptr = c_void_p()

# 			# 	if not err:
# 			# 		LLT_ptr = LLT.ctypes.data_as(lib.ok_float_p)
# 			# 	else:
# 			# 		LLT_ptr = None


# 			# 	if not err and 'd' in data:
# 			# 		d = data['d'].astype(lib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'd.npy')):
# 			# 		d = np_load(path.join(directory, 'd.npy')).astype(
# 			# 					lib.pyfloat)
# 			# 	else:
# 			# 		err = 1

# 			# 	if not err and 'e' in data:
# 			# 		e = data['e'].astype(lib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'e.npy')):
# 			# 		e = np_load(path.join(directory, 'e.npy')).astype(
# 			# 					lib.pyfloat)
# 			# 	else:
# 			# 		err = 1

# 			# 	if err:
# 			# 		snippet = '`LLT`, ' iflib.direct else ''
# 			# 		ValueError('Minimal requirements to load solver '
# 			# 				   'not met. Specified file must contain '
# 			# 				   'at least one .npz file with entries `A_equil`,'
# 			# 				   ' {}`d`, and `e`, or the specified folder must'
# 			# 				   ' contain .npy files of the same names.'.format(
# 			# 				   snippet))

# 			# 	if 'z' in data:
# 			# 		z = data['z'].astype(lib.pyfloat)
# 			# 	else:
# 			# 		z = zeros(self.m + self.n, dtype=lib.pyfloat)

# 			# 	if 'z12' in data:
# 			# 		z12 = data['z12'].astype(lib.pyfloat)
# 			# 	else:
# 			# 		z12 = zeros(self.m + self.n, dtype=lib.pyfloat)

# 			# 	if 'zt' in data:
# 			# 		zt = data['zt'].astype(lib.pyfloat)
# 			# 	else:
# 			# 		zt = zeros(self.m + self.n, dtype=lib.pyfloat)

# 			# 	if 'zt12' in data:
# 			# 		zt12 = data['zt12'].astype(lib.pyfloat)
# 			# 	else:
# 			# 		zt12 = zeros(self.m + self.n, dtype=lib.pyfloat)

# 			# 	if 'zprev' in data:
# 			# 		zprev = data['zprev'].astype(lib.pyfloat)
# 			# 	else:
# 			# 		zprev = zeros(self.m + self.n, dtype=lib.pyfloat)

# 			# 	if 'rho' in data:
# 			# 		rho = lib.pyfloat(data['rho'])
# 			# 	else:
# 			# 		rho = 1.

# 			# 	order = lib.enums.CblasRowMajor if \
# 			# 		A_equil.flags.c_contiguous else lib.enums.CblasColMajor

# 			# 	if self.c_solver is not None:
# 			# 	lib.pogs_finish(self.c_solver)

# 			# 	self.c_solver =lib.pogs_load_solver(
# 			# 			A_equil.ctypes.data_as(lib.ok_float_p), LLT_ptr,
# 			# 			d.ctypes.data_as(lib.ok_float_p),
# 			# 			e.ctypes.data_as(lib.ok_float_p),
# 			# 			z.ctypes.data_as(lib.ok_float_p),
# 			# 			z12.ctypes.data_as(lib.ok_float_p),
# 			# 			zt.ctypes.data_as(lib.ok_float_p),
# 			# 			zt12.ctypes.data_as(lib.ok_float_p),
# 			# 			zprev.ctypes.data_as(lib.ok_float_p),
# 			# 			rho, self.m, self.n, order)

# 			# def save(self, directory, name, save_equil=True,
# 			# 		 save_factorization=True):

# 			# 	if self.c_solver is None:
# 			# 		raise ValueError(
# 			# 				'No solver intialized, save() call invalid')

# 			# 	filename = path.join(directory, name)
# 			# 	if not name.endswith('.npz'):
# 			# 		filename.join('.npz')

# 			# 	if not path.exists(directory):
# 			# 		raise ValueError('specified directory does not exist')

# 			# 	if path.exists(filename):
# 			# 		raise ValueError('specified filepath already exists '
# 			# 						 'and would be overwritten, aborting.')

# 			# 	mindim = min(self.m, self.n)
# 			# 	A_equil = zeros((self.m, self.n), dtype=lib.pyfloat)

# 			# 	iflib.direct:
# 			# 		LLT = zeros((mindim, mindim), dtype=lib.pyfloat)
# 			# 		LLT_ptr = LLT.ctypes.data_as(lib.ok_float_p)
# 			# 	else:
# 			# 		LLT = c_void_p()
# 			# 		LLT_ptr = LLT

# 			# 	if A_equil.flags.c_contiguous:
# 			# 		order = lib.enums.CblasRowMajor
# 			# 	else:
# 			# 		order = lib.enums.CblasColMajor

# 			# 	d = zeros(self.m, dtype=lib.pyfloat)
# 			# 	e = zeros(self.n, dtype=lib.pyfloat)
# 			# 	z = zeros(self.m + self.n, dtype=lib.pyfloat)
# 			# 	z12 = zeros(self.m + self.n, dtype=lib.pyfloat)
# 			# 	zt = zeros(self.m + self.n, dtype=lib.pyfloat)
# 			# 	zt12 = zeros(self.m + self.n, dtype=lib.pyfloat)
# 			# 	zprev = zeros(self.m + self.n, dtype=lib.pyfloat)
# 			# 	rho = zeros(1, dtype=lib.pyfloat)

# 			# lib.pogs_extract_solver(
# 			# 			self.c_solver,
# 			# 			A_equil.ctypes.data_as(lib.ok_float_p), LLT_ptr,
# 			# 			d.ctypes.data_as(lib.ok_float_p),
# 			# 			e.ctypes.data_as(lib.ok_float_p),
# 			# 			z.ctypes.data_as(lib.ok_float_p),
# 			# 			z12.ctypes.data_as(lib.ok_float_p),
# 			# 			zt.ctypes.data_as(lib.ok_float_p),
# 			# 			zt12.ctypes.data_as(lib.ok_float_p),
# 			# 			zprev.ctypes.data_as(lib.ok_float_p),
# 			# 			rho.ctypes.data_as(lib.ok_float_p), order)

# 			# 	if isinstance(LLT, ndarray) and save_factorization:
# 			# 		savez(filename, A_equil=A_equil, LLT=LLT, d=d, e=e, z=z,
# 			# 			  z12=z12, zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
# 			# 	elif save_equil:
# 			# 		savez(filename, A_equil=A_equil, d=d, e=e, z=z, z12=z12,
# 			# 			  zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
# 			# 	else:
# 			# 		savez(filename, z=z, z12=z12, zt=zt, zt12=zt12,
# 			# 			  zprev=zprev, rho=rho[0])

		self.Solver = Solver