from ctypes import c_void_p, byref
from optkit.libs import *
from optkit.types import LowLevelTypes
from optkit.utils import UtilMakeCVector, UtilMakeCMatrix, \
	UtilMakeCFunctionVector, UtilReleaseCVector, UtilReleaseCMatrix, \
	UtilReleaseCFunctionVector
from os import getenv

# CPU32, CPU64, GPU32, GPU64
class OKBackend(object):
	def __init__(self, GPU=False, single_precision=False):

		self.device = None
		self.precision = None
		self.libname = "(No libraries selected)"

		self.lowtypes = None
		self.dimcheck = getenv('OPTKIT_CHECKDIM', 0)
		self.typecheck = getenv('OPTKIT_CHECKTYPE', 0)
		self.devicecheck = getenv('OPTKIT_CHECKDEVICE', 0)

		# library loaders
		self.dense_lib_loader = DenseLinsysLibs()
		self.sparse_lib_loader = None
		self.prox_lib_loader = ProxLibs()

		# library instances
		self.dense = None
		self.sparse = None
		self.prox = None

		self.dense_blas_handle = c_void_p()
		self.sparse_blas_handle = None

		self.__LIBGUARD_ON__ = False
		self.__set_lib()

		if self.dense is not None: self.dense.blas_make_handle(byref(self.dense_blas_handle))
		if self.sparse is not None: self.sparse.blas_make_handle(self.sparse_blas_handle)

	def __set_lib(self, device=None, precision=None, order=None):
		if self.__LIBGUARD_ON__:
			print str('Backend cannot be changed once ' 
				'Vector/Matrix/FunctionVector objects have been created.\n')
		devices = ['gpu', 'cpu'] if device == 'gpu' else ['cpu', 'gpu']
		precisions = ['32', '64'] if precision == '32' else ['64', '32']
		orders = ['col', ''] if order == 'col' else ['row', ''] \
			if order == 'row' else ['']


 		for dev in devices:
			for prec in precisions:
				for layout in orders:
					lib_key = layout
					if lib_key != '': lib_key += '_'
					lib_key +='{}{}'.format(dev, prec)
					lib_key_prox = '{}{}'.format(dev, prec)
					valid = self.dense_lib_loader.libs[lib_key] is not None
					valid &= self.prox_lib_loader.libs[lib_key_prox] is not None
					if self.sparse_lib_loader is not None:
						valid &= self.sparse_lib_loader.libs[lib_key] is not None
					if valid:
						self.lowtypes = LowLevelTypes(
							single_precision = prec=='32', order=layout)
						self.device = dev
						self.precision = prec
						self.layout = layout
						self.libname = lib_key 
						self.dense = self.dense_lib_loader.get(
							self.lowtypes, GPU=dev=='gpu')
						self.prox = self.prox_lib_loader.get(
							self.lowtypes, GPU=dev=='gpu')
						if self.sparse_lib_loader is not None:
							self.sparse = self.sparse_lib_loader.get(
								self.lowtypes, GPU=dev=='gpu')


						self.make_cvector = UtilMakeCVector(self.lowtypes, self.dense)
						self.release_cvector = UtilReleaseCVector(self.lowtypes, self.dense)
						self.make_cmatrix = UtilMakeCMatrix(self.lowtypes, self.dense)
						self.release_cmatrix = UtilReleaseCMatrix(self.lowtypes, self.dense)
						self.make_cfunctionvector = UtilMakeCFunctionVector(self.lowtypes, self.prox)
						self.release_cfunctionvector = UtilReleaseCFunctionVector(self.lowtypes, self.prox)

						return 
					else:
						print str('Libraries for configuration {} '
							'not found. Trying next layout/device/precision '
							'configuration.'.format(lib_key))
		raise RuntimeError("No libraries found for backend.")

	def change(self, GPU=False, double=True, 
		force_rowmajor=False, force_colmajor=False,
		checktypes=None, checkdims=None, checkdevices=None):

		precision = '64' if double else '32'
		device = 'gpu' if GPU else 'cpu'
		order = 'row' if force_rowmajor else 'col' if force_colmajor else '' 

		if self.dense is not None: self.dense.blas_destroy_handle(self.dense_blas_handle)
		if self.sparse is not None: self.sparse.blas_destroy_handle(self.sparse_blas_handle)

		self.__set_lib(device=device, precision=precision, order=order)
		

		if self.dense is not None: self.dense.blas_make_handle(byref(self.dense_blas_handle))
		if self.sparse is not None: self.sparse.blas_make_handle(byref(self.sparse_blas_handle))

		if checktypes is not None: self.typecheck = checktypes
		if checkdims is not None: self.dimcheck = checkdims
		if checkdevices is not None: self.devicecheck = checkdevices


	def __del__(self):
		if self.dense is not None: self.dense.blas_destroy_handle(self.dense_blas_handle)
		if self.sparse is not None: self.sparse.blas_destroy_handle(self.sparse_blas_handle)