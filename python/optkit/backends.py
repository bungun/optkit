from os import getenv
import gc
from ctypes import c_int, c_void_p, byref, pointer, CDLL
# from optkit.libs.linsys import DenseLinsysLibs, SparseLinsysLibs
# from optkit.libs.prox import ProxLibs
from optkit.libs.pogs import PogsLibs
from optkit.libs.clustering import ClusteringLibs
from optkit.utils.pyutils import version_string


# CPU32, CPU64, GPU32, GPU64
class OKBackend(object):
	def __init__(self, gpu=False, single_precision=False):
		self.version = None
		self.__device = None
		self.__precision = None
		self.__config = "(No libraries selected)"

		self.lowtypes = None
		self.dimcheck = getenv('OPTKIT_CHECKDIM', 0)
		self.typecheck = getenv('OPTKIT_CHECKTYPE', 0)
		self.devicecheck = getenv('OPTKIT_CHECKDEVICE', 0)

		# library loaders
		# self.dense_lib_loader = DenseLinsysLibs()
		# self.sparse_lib_loader = SparseLinsysLibs()
		# self.prox_lib_loader = ProxLibs()
		self.pogs_lib_loader = PogsLibs()
		self.cluster_lib_loader = ClusteringLibs()

		# library instances
		# self.dense = None
		# self.sparse = None
		# self.prox = None
		self.pogs = None
		self.cluster = None

		self.__LIBGUARD_ON = False
		self.__COBJECT_COUNT = 0
		self.__set_lib()

	@property
	def config(self):
		return self.__config

	@property
	def precision(self):
		return self.__precision

	@property
	def precision_is_32bit(self):
		return self.__precision == '32'
	@property
	def device(self):
		return self.__device

	@property
	def device_is_gpu(self):
		return self.__device == 'gpu'

	@property
	def device_reset_allowed(self):
		return self.__COBJECT_COUNT == 0

	@property
	def libguard_active(self):
		return self.__LIBGUARD_ON

	def increment_cobject_count(self):
		self.__COBJECT_COUNT += 1
		self.__LIBGUARD_ON = True

	def decrement_cobject_count(self):
		self.__COBJECT_COUNT -= 1
		self.__LIBGUARD_ON = self.__COBJECT_COUNT == 0

	def __get_version(self):
		major = c_int()
		minor = c_int()
		change = c_int()
		status = c_int()
		try:
			self.pogs.optkit_version(byref(major), byref(minor), byref(change),
									 byref(status))

			self.version = "Optkit v{}".format(
					version_string(major.value, minor.value, change.value,
								   status.value))
		except:
			self.version = "Optkit: version unknown"

	def load_lib(self, name, override=False):
		if name not in ['pogs', 'cluster']:
			raise ValueError('invalid library name')
		elif self.__dict__[name] is not None:
			if not override:
				print str('\nlibrary {} already loaded; call with keyword arg '
					  '"override"=True to bypass this check\n'.format(name))

		if name == 'pogs':
			self.pogs = self.pogs_lib_loader.get(
					single_precision=self.precision_is_32bit,
					gpu=self.device_is_gpu)
		elif name == 'cluster':
			self.cluster = self.cluster_lib_loader.get(
					single_precision=self.precision_is_32bit,
					gpu=self.device_is_gpu)

	def load_libs(self, *names):
		for name in names:
			self.load_lib(name)

	def __set_lib(self, device=None, precision=None, order=None):
		devices = ['gpu', 'cpu'] if device == 'gpu' else ['cpu', 'gpu']
		precisions = ['32', '64'] if precision == '32' else ['64', '32']

 		for dev in devices:
			for prec in precisions:
				lib_key ='{}{}'.format(dev, prec)
				# lib_key_prox = '{}{}'.format(dev, prec)
				# valid = self.dense_lib_loader.libs[lib_key] is not None
				# valid &= self.prox_lib_loader.libs[lib_key] is not None
				# valid &= self.sparse_lib_loader.libs[lib_key] is not None
				valid = self.pogs_lib_loader.libs[lib_key] is not None
				if valid:
					self.__device = dev
					self.__precision = prec
					self.__config = lib_key
					self.load_lib('pogs', override=True)
					self.load_lib('cluster', override=True)
					# self.dense = self.dense_lib_loader.get(
							# single_precision=single, gpu=gpu)
					# self.prox = self.prox_lib_loader.get(
							# single_precision=single, gpu=gpu)
					# self.sparse = self.sparse_lib_loader.get(
							# single_precision=single, gpu=gpu)
					return
				else:
					print str('Libraries for configuration {} '
							  'not found. Trying next configuration.'.format(
							  lib_key))

		raise RuntimeError('No libraries found for backend.')

	def change(self, gpu=False, double=True, checktypes=None, checkdims=None,
			   checkdevices=None):

		if self.__LIBGUARD_ON:
			print str('Backend cannot be changed once C objects have been '
					  'created.\n')
			return

		precision = '64' if double else '32'
		device = 'gpu' if gpu else 'cpu'

		self.__set_lib(device=device, precision=precision)
		self.__get_version()

		if checktypes is not None: self.typecheck = checktypes
		if checkdims is not None: self.dimcheck = checkdims
		if checkdevices is not None: self.devicecheck = checkdevices

	def reset_device(self):
		if self.device_reset_allowed:
			for item in self.__dict__:
				if isinstance(item, CDLL):
					if 'ok_device_reset' in item.__dict__:
						if self.pogs.ok_device_reset():
							raise RuntimeError('device reset failed')
						return
			raise RuntimeError('device reset not possible: '
							   'no libraries loaded')
		else:
			raise RuntimeError('device reset not allowed: '
							   'C objects allocated')