# from numpy import zeros, ones, ndarray, savez, load as np_load
# from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
# from ctypes import c_void_p
# from os import path
# from optkit.types.operator import OperatorTypes
# from optkit.types.pogs.common import PogsTypes

# class PogsOperatorTypes(PogsTypes):
# 	def __init__(self, backend):
# 		PogsTypes.__init__(self, backend)
# 		PogsSettings = backend.pogs.pogs_settings
# 		PogsInfo = backend.pogs.pogs_info
# 		PogsOutput = backend.pogs.pogs_output
# 		pogslib = backend.pogs
# 		denselib = backend.dense
# 		sparselib = backend.sparse
# 		proxlib = backend.prox
# 		operatorlib = backend.operator

# 		Operator = OperatorTypes(backend).AbstractLinearOperator

# 		class Solver(object):
# 			def __init__(self, A, **options):
# 				self.A = Operator(A)
# 				self.shape = (self.m, self.n) = (m, n) = self.A.shape
# 				self.__f = zeros(m).astype(proxlib.function)
# 				self.__f_ptr = self.__f.ctypes.data_as(proxlib.function_p)
# 				self.__f_c = proxlib.function_vector(self.m, self.__f_ptr)
# 				self.__g = zeros(n).astype(proxlib.function)
# 				self.__g_ptr = self.__g.ctypes.data_as(proxlib.function_p)
# 				self.__g_c = proxlib.function_vector(self.n, self.__g_ptr)
# 				self.layout = layout = denselib.enums.CblasRowMajor if \
# 					A.flags.c_contiguous else denselib.enums.CblasColMajor

# 				NO_INIT = bool(options.pop('no_init', False))
# 				DIRECT = int(options.pop('direct', False))
# 				EQUILNORM = float(options.pop('equil_norm', 1.))

# 				if NO_INIT:
# 					self.c_solver = pogslib.pogs_init(self.A.c_ptr, DIRECT,
# 													  EQUILNORM)
# 					backend.increment_cobject_count()
# 				else:
# 					self.c_solver = None

# 				self.settings = SolverSettings()
# 				self.info = SolverInfo()
# 				self.output = SolverOutput(m, n)
# 				self.settings.update(**options)
# 				self.first_run = True

# 			def __update_function_vectors(self, f, g):
# 				for i in xrange(f.size):
# 					self.__f[i] = proxlib.function(f.h[i], f.a[i], f.b[i],
# 												   f.c[i], f.d[i], f.e[i])

# 				for j in xrange(g.size):
# 					self.__g[j] = proxlib.function(g.h[j], g.a[j], g.b[j],
# 												   g.c[j], g.d[j], g.e[j])

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
# 				pogslib.pogs_solve(self.c_solver, self.__f_c, self.__g_c,
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
# 			# 		A_equil = data['A_equil'].astype(denselib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'A_equil.npy')):
# 			# 		A_equil = np_load(path.join(directory,
# 			# 			'A_equil.npy')).astype(denselib.pyfloat)
# 			# 	else:
# 			# 		err = 1


# 			# 	if not err and 'LLT' in data:
# 			# 		LLT = data['LLT'].astype(denselib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'LLT.npy')):
# 			# 		LLT = np_load(path.join(directory, 'LLT.npy')).astype(
# 			# 					denselib.pyfloat)
# 			# 		LLT_ptr = LLT.ctypes.data_as(denselib.ok_float_p)
# 			# 	else:
# 			# 		if pogslib.direct:
# 			# 			err = 1
# 			# 		else:
# 			# 			LLT_ptr = c_void_p()

# 			# 	if not err:
# 			# 		LLT_ptr = LLT.ctypes.data_as(denselib.ok_float_p)
# 			# 	else:
# 			# 		LLT_ptr = None


# 			# 	if not err and 'd' in data:
# 			# 		d = data['d'].astype(denselib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'd.npy')):
# 			# 		d = np_load(path.join(directory, 'd.npy')).astype(
# 			# 					denselib.pyfloat)
# 			# 	else:
# 			# 		err = 1

# 			# 	if not err and 'e' in data:
# 			# 		e = data['e'].astype(denselib.pyfloat)
# 			# 	elif path.exists(path.join(directory, 'e.npy')):
# 			# 		e = np_load(path.join(directory, 'e.npy')).astype(
# 			# 					denselib.pyfloat)
# 			# 	else:
# 			# 		err = 1

# 			# 	if err:
# 			# 		snippet = '`LLT`, ' if pogslib.direct else ''
# 			# 		ValueError('Minimal requirements to load solver '
# 			# 				   'not met. Specified file must contain '
# 			# 				   'at least one .npz file with entries `A_equil`,'
# 			# 				   ' {}`d`, and `e`, or the specified folder must'
# 			# 				   ' contain .npy files of the same names.'.format(
# 			# 				   snippet))

# 			# 	if 'z' in data:
# 			# 		z = data['z'].astype(denselib.pyfloat)
# 			# 	else:
# 			# 		z = zeros(self.m + self.n, dtype=denselib.pyfloat)

# 			# 	if 'z12' in data:
# 			# 		z12 = data['z12'].astype(denselib.pyfloat)
# 			# 	else:
# 			# 		z12 = zeros(self.m + self.n, dtype=denselib.pyfloat)

# 			# 	if 'zt' in data:
# 			# 		zt = data['zt'].astype(denselib.pyfloat)
# 			# 	else:
# 			# 		zt = zeros(self.m + self.n, dtype=denselib.pyfloat)

# 			# 	if 'zt12' in data:
# 			# 		zt12 = data['zt12'].astype(denselib.pyfloat)
# 			# 	else:
# 			# 		zt12 = zeros(self.m + self.n, dtype=denselib.pyfloat)

# 			# 	if 'zprev' in data:
# 			# 		zprev = data['zprev'].astype(denselib.pyfloat)
# 			# 	else:
# 			# 		zprev = zeros(self.m + self.n, dtype=denselib.pyfloat)

# 			# 	if 'rho' in data:
# 			# 		rho = denselib.pyfloat(data['rho'])
# 			# 	else:
# 			# 		rho = 1.

# 			# 	order = denselib.enums.CblasRowMajor if \
# 			# 		A_equil.flags.c_contiguous else denselib.enums.CblasColMajor

# 			# 	if self.c_solver is not None:
# 			# 		pogslib.pogs_finish(self.c_solver)

# 			# 	self.c_solver = pogslib.pogs_load_solver(
# 			# 			A_equil.ctypes.data_as(denselib.ok_float_p), LLT_ptr,
# 			# 			d.ctypes.data_as(denselib.ok_float_p),
# 			# 			e.ctypes.data_as(denselib.ok_float_p),
# 			# 			z.ctypes.data_as(denselib.ok_float_p),
# 			# 			z12.ctypes.data_as(denselib.ok_float_p),
# 			# 			zt.ctypes.data_as(denselib.ok_float_p),
# 			# 			zt12.ctypes.data_as(denselib.ok_float_p),
# 			# 			zprev.ctypes.data_as(denselib.ok_float_p),
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
# 			# 	A_equil = zeros((self.m, self.n), dtype=denselib.pyfloat)

# 			# 	if pogslib.direct:
# 			# 		LLT = zeros((mindim, mindim), dtype=denselib.pyfloat)
# 			# 		LLT_ptr = LLT.ctypes.data_as(denselib.ok_float_p)
# 			# 	else:
# 			# 		LLT = c_void_p()
# 			# 		LLT_ptr = LLT

# 			# 	if A_equil.flags.c_contiguous:
# 			# 		order = denselib.enums.CblasRowMajor
# 			# 	else:
# 			# 		order = denselib.enums.CblasColMajor

# 			# 	d = zeros(self.m, dtype=denselib.pyfloat)
# 			# 	e = zeros(self.n, dtype=denselib.pyfloat)
# 			# 	z = zeros(self.m + self.n, dtype=denselib.pyfloat)
# 			# 	z12 = zeros(self.m + self.n, dtype=denselib.pyfloat)
# 			# 	zt = zeros(self.m + self.n, dtype=denselib.pyfloat)
# 			# 	zt12 = zeros(self.m + self.n, dtype=denselib.pyfloat)
# 			# 	zprev = zeros(self.m + self.n, dtype=denselib.pyfloat)
# 			# 	rho = zeros(1, dtype=denselib.pyfloat)

# 			# 	pogslib.pogs_extract_solver(
# 			# 			self.c_solver,
# 			# 			A_equil.ctypes.data_as(denselib.ok_float_p), LLT_ptr,
# 			# 			d.ctypes.data_as(denselib.ok_float_p),
# 			# 			e.ctypes.data_as(denselib.ok_float_p),
# 			# 			z.ctypes.data_as(denselib.ok_float_p),
# 			# 			z12.ctypes.data_as(denselib.ok_float_p),
# 			# 			zt.ctypes.data_as(denselib.ok_float_p),
# 			# 			zt12.ctypes.data_as(denselib.ok_float_p),
# 			# 			zprev.ctypes.data_as(denselib.ok_float_p),
# 			# 			rho.ctypes.data_as(denselib.ok_float_p), order)

# 			# 	if isinstance(LLT, ndarray) and save_factorization:
# 			# 		savez(filename, A_equil=A_equil, LLT=LLT, d=d, e=e, z=z,
# 			# 			  z12=z12, zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
# 			# 	elif save_equil:
# 			# 		savez(filename, A_equil=A_equil, d=d, e=e, z=z, z12=z12,
# 			# 			  zt=zt, zt12=zt12, zprev=zprev, rho=rho[0])
# 			# 	else:
# 			# 		savez(filename, z=z, z12=z12, zt=zt, zt12=zt12,
# 			# 			  zprev=zprev, rho=rho[0])

# 			def __del__(self):
# 				if self.c_solver is not None:
# 					pogslib.pogs_finish(self.c_solver, 0)
# 					backend.decrement_cobject_count()

# 		self.Solver = Solver