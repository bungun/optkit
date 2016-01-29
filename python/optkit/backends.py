from ctypes import c_int, c_void_p, byref, pointer
from optkit.libs import *
from optkit.types import LowLevelTypes
from optkit.utils import UtilMakeCVector, UtilMakeCMatrix, \
	UtilMakeCSparseMatrix, UtilMakeCFunctionVector, \
	UtilReleaseCVector, UtilReleaseCMatrix, \
	UtilReleaseCSparseMatrix, UtilReleaseCFunctionVector
from optkit.utils.pyutils import version_string
from os import getenv
import gc

# CPU32, CPU64, GPU32, GPU64
class OKBackend(object):
	def __init__(self, GPU=False, single_precision=False):
		self.version = None
		self.device = None
		self.precision = None
		self.libname = "(No libraries selected)"

		self.lowtypes = None
		self.dimcheck = getenv('OPTKIT_CHECKDIM', 0)
		self.typecheck = getenv('OPTKIT_CHECKTYPE', 0)
		self.devicecheck = getenv('OPTKIT_CHECKDEVICE', 0)

		# library loaders
		self.dense_lib_loader = DenseLinsysLibs()
		self.sparse_lib_loader = SparseLinsysLibs()
		self.prox_lib_loader = ProxLibs()
		self.pogs_lib_loader = PogsLibs(dense=True)

		# library instances
		self.dense = None
		self.sparse = None
		self.prox = None
		self.pogs = None

		self.dense_blas_handle = c_void_p(0)
		self.sparse_handle = c_void_p(0)

		self.__LIBGUARD_ON__ = False
		self.__HANDLES_MADE__ = False
		self.__CSOLVER_COUNT__ = 0
		self.__COBJECT_COUNT__ = 0
		self.__set_lib()

		self.make_linalg_contexts()

	def reset(self):
		self.__LIBGUARD_ON__ = False
		self.destroy_linalg_contexts()

	def make_linalg_contexts(self):
		if not self.__HANDLES_MADE__:
			self.destroy_linalg_contexts()
			if self.dense is not None: self.dense.blas_make_handle(byref(self.dense_blas_handle))
			if self.sparse is not None: self.sparse.sp_make_handle(self.sparse_handle)
			self.__HANDLES_MADE__ = True

			self.make_cvector = UtilMakeCVector(self.lowtypes, self.dense)
			self.release_cvector = UtilReleaseCVector(self.lowtypes, self.dense)
			self.make_cmatrix = UtilMakeCMatrix(self.lowtypes, self.dense)
			self.release_cmatrix = UtilReleaseCMatrix(self.lowtypes, self.dense)
			self.make_csparsematrix = UtilMakeCSparseMatrix(self.sparse_handle, self.lowtypes, self.sparse)
			self.release_csparsematrix = UtilReleaseCSparseMatrix(self.lowtypes, self.sparse)
			self.make_cfunctionvector = UtilMakeCFunctionVector(self.lowtypes, self.prox)
			self.release_cfunctionvector = UtilReleaseCFunctionVector(self.lowtypes, self.prox)


	def destroy_linalg_contexts(self):
		if self.__COBJECT_COUNT__ > 0:
			gc.collect()
			if self.__COBJECT_COUNT__ > 0:
				RuntimeError(str("All optkit objects "
					"(Vector, Matrix, SparseMatrix, FunctionVector) "
					"must be out of scope to reset backend "
					"linear algebra contexts"))

		if self.__HANDLES_MADE__:
			if self.dense is not None: 
				self.dense.blas_destroy_handle(self.dense_blas_handle)
			if self.sparse is not None: 
				self.sparse.sp_destroy_handle(self.sparse_handle)
			self.__HANDLES_MADE__ = False
		if self.device_reset_allowed:
			self.dense.ok_device_reset()

		self.make_cvector = None 
		self.release_cvector = None
		self.make_cmatrix = None
		self.release_cmatrix = None
		self.make_csparsematrix = None
		self.release_csparsematrix = None
		self.make_cfunctionvector = None
		self.release_cfunctionvector = None


	@property	
	def device_reset_allowed(self):
		return self.__CSOLVER_COUNT__ == 0 and not self.__HANDLES_MADE__

	@property
	def linalg_contexts_exist(self):
		return self.__HANDLES_MADE__

	@property
	def libguard_active(self):
		return self.__LIBGUARD_ON__

	def increment_cobject_count(self):
		self.__COBJECT_COUNT__ += 1
		self.__LIBGUARD_ON__ = True

	def decrement_cobject_count(self):
		self.__COBJECT_COUNT__ -= 1
		self.__LIBGUARD_ON__ = self.__COBJECT_COUNT__ == 0

	def increment_csolver_count(self):
		self.__CSOLVER_COUNT__ += 1

	def decrement_csolver_count(self):
		self.__CSOLVER_COUNT__ -= 1


	def __get_version(self):
		major = c_int()
		minor = c_int()
		change = c_int()
		status = c_int()
		try:
			self.dense.denselib_version(byref(major), byref(minor),
				byref(change), byref(status))
			self.version = "Optkit v{}".format(version_string(
				major.value, minor.value, change.value, status.value))
		except:
			self.version = "Optkit: version unknown"

	def __set_lib(self, device=None, precision=None, order=None):
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
					valid &= self.sparse_lib_loader.libs[lib_key] is not None
					valid &= self.pogs_lib_loader.libs[lib_key] is not None
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
						self.sparse = self.sparse_lib_loader.get(
							self.lowtypes, GPU=dev=='gpu')
						self.pogs = self.pogs_lib_loader.get(
							self.lowtypes, GPU=dev=='gpu')
						
						return 
					else:
						print str('Libraries for configuration {} '
							'not found. Trying next layout/device/precision '
							'configuration.'.format(lib_key))


		raise RuntimeError("No libraries found for backend.")

	def change(self, GPU=False, double=True, 
		force_rowmajor=False, force_colmajor=False,
		checktypes=None, checkdims=None, checkdevices=None):

		if self.__LIBGUARD_ON__:
			print str('Backend cannot be changed once ' 
				'Vector/Matrix/FunctionVector objects have been created.\n')
			return

		precision = '64' if double else '32'
		device = 'gpu' if GPU else 'cpu'
		order = 'row' if force_rowmajor else 'col' if force_colmajor else '' 

		self.destroy_linalg_contexts()
		self.__set_lib(device=device, precision=precision, order=order)
		self.__get_version()
		
		if checktypes is not None: self.typecheck = checktypes
		if checkdims is not None: self.dimcheck = checkdims
		if checkdevices is not None: self.devicecheck = checkdevices

	def __del__(self):
		self.destroy_linalg_contexts()