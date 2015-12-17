from ctypes import c_void_p
from optkit.libs import *
from optkit.types.lowlevel import ok_lowtypes
from optkit.utils import UtilMakeCVector, UtilMakeCMatrix, \
	UtilMakeCFunctionVector, UtilReleaseCVector, UtilReleaseCMatrix, \
	UtilReleaseCFunctionVector


# CPU32, CPU64, GPU32, GPU64
class OKBackend(object):
	def __init__(self, GPU=False, single_precision=False, 
		checktypes=True, checkdims=True):

		self.device = None
		self.precision = None

		self.lowtypes = ok_lowtypes
		self.dimcheck_default = checkdims
		self.typecheck_default = checktypes

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

		self.__set_lib()

		if self.dense is not None: self.dense.blas_make_handle(self.dense_blas_handle)
		if self.sparse is not None: self.sparse.blas_make_handle(self.sparse_blas_handle)

	def __set_lib(self, device=None, precision=None):
		devices = ['gpu', 'cpu'] if device=='gpu' else ['cpu', 'gpu']
		precisions = ['32', '64'] if precision=='32' else ['64', '32']

 		for dev in devices:
			for prec in precisions:
				lib_key = '{}{}'.format(dev, prec)
				valid = self.dense_lib_loader.libs.has_key(lib_key)
				valid &= self.prox_lib_loader.libs.has_key(lib_key)
				if self.sparse_lib_loader is not None:
					valid &= self.sparse_lib_loader.libs.has_key(lib_key)
				if valid:
					self.lowtypes.change_precision(single_precision=prec=='32')
					self.device=dev
					self.precision=prec
					self.dense = self.dense_lib_loader.get(self.lowtypes, GPU=dev=='gpu')
					self.prox = self.prox_lib_loader.get(self.lowtypes, GPU=dev=='gpu')
					if self.sparse_lib_loader is not None:
						self.sparse = self.sparse_lib_loader.get(self.lowtypes, GPU=dev=='gpu')


					self.make_cvector = UtilMakeCVector(self.lowtypes, self.dense)
					self.release_cvector = UtilReleaseCVector(self.lowtypes, self.dense)
					self.make_cmatrix = UtilMakeCMatrix(self.lowtypes, self.dense)
					self.release_cmatrix = UtilReleaseCMatrix(self.lowtypes, self.dense)
					self.make_cfunctionvector = UtilMakeCFunctionVector(self.lowtypes, self.prox)
					self.release_cfunctionvector = UtilReleaseCFunctionVector(self.lowtypes, self.prox)

					return
				else:
					print str('Libraries for configuration {} '
						'not found. Trying next device/precision '
						'configuration.'.format(lib_key))
		raise RuntimeError("No libraries found for backend.")

	def change(self, GPU=False, single_precision=False, 
		checktypes=True, checkdims=True):
		precision = '32' if single_precision else '64'
		device = 'gpu' if GPU else 'cpu'

		if self.dense is not None: self.dense.blas_destroy_handle(self.dense_blas_handle)
		if self.sparse is not None: self.sparse.blas_destroy_handle(self.sparse_blas_handle)

		self.__set_lib(device=device, precision=precision)

		if self.dense is not None: self.dense.blas_make_handle(self.dense_blas_handle)
		if self.sparse is not None: self.sparse.blas_make_handle(self.sparse_blas_handle)


	def __del__(self):
		if self.dense is not None: self.dense.blas_destroy_handle(self.dense_blas_handle)
		if self.sparse is not None: self.sparse.blas_destroy_handle(self.sparse_blas_handle)