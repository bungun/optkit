from optkit.compat import *

import gc
import ctypes as ct

# from optkit.libs.linsys import DenseLinsysLibs, SparseLinsysLibs
# from optkit.libs.prox import ProxLibs
from optkit.libs.pogs import PogsLibs
from optkit.libs.clustering import ClusteringLibs
from optkit.utils.pyutils import version_string


# CPU32, CPU64, GPU32, GPU64
class OKBackend(object):
	__LIBNAMES = ['pogs', 'cluster']

	def __init__(self, gpu=False, single_precision=False):
		self.__version = None
		self.__device = None
		self.__precision = None
		self.__config = None

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
		self.__active_libs = []

		self.__set_lib()

	def __clear(self):
		self.__version = None
		self.__device = None
		self.__precision = None
		self.__config = '(No libraries selected)'

		# library instances
		# self.dense = None
		# self.sparse = None
		# self.prox = None
		self.pogs = None
		self.cluster = None

		self.__LIBGUARD_ON = False
		self.__COBJECT_COUNT = 0

	@property
	def version(self):
		return self.__version

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
		self.__LIBGUARD_ON = self.__COBJECT_COUNT > 0

	def __get_version(self):
		major = ct.c_int()
		minor = ct.c_int()
		change = ct.c_int()
		status = ct.c_int()
		try:
			self.pogs.optkit_version(
					ct.byref(major), ct.byref(minor), ct.byref(change),
					ct.byref(status))

			self.__version = 'Optkit v{}'.format(
					version_string(major.value, minor.value, change.value,
								   status.value))
		except:
			self.__version = 'Optkit: version unknown'

	def load_lib(self, name, override=False):
		if name not in self.__LIBNAMES:
			raise ValueError('invalid library name')
		elif self.__dict__[name] is not None:
			if not override:
				print('\nlibrary {} already loaded; call with keyword arg '
					  '"override"=True to bypass this check\n'.format(name))

		if name == 'pogs':
			self.pogs = self.pogs_lib_loader.get(
					single_precision=self.precision_is_32bit,
					gpu=self.device_is_gpu)
			self.__active_libs.append(self.pogs)
		elif name == 'cluster':
			self.cluster = self.cluster_lib_loader.get(
					single_precision=self.precision_is_32bit,
					gpu=self.device_is_gpu)
			self.__active_libs.append(self.cluster)


	def load_libs(self, *names):
		for name in names:
			self.load_lib(name)

	def __set_lib(self, device=None, precision=None, order=None):
		self.__clear()

		devices = ['gpu', 'cpu'] if device == 'gpu' else ['cpu', 'gpu']
		precisions = ['32', '64'] if precision == '32' else ['64', '32']

		for dev in devices:
			for prec in precisions:
				lib_key ='{}{}'.format(dev, prec)
				valid = self.pogs_lib_loader.libs[lib_key] is not None
				if valid:
					self.__device = dev
					self.__precision = prec
					self.__config = lib_key
					self.load_lib('pogs', override=True)
					self.load_lib('cluster', override=True)
					return
				else:
					print ('Libraries for configuration {} '
						   'not found. Trying next configuration.'.format(
						   	lib_key))

		raise RuntimeError('No libraries found for backend.')

	def change(self, gpu=False, double=True):
		if self.__LIBGUARD_ON:
			print('Backend cannot be changed once C objects have been '
				  'created.\n')
			return

		precision = '64' if double else '32'
		device = 'gpu' if gpu else 'cpu'

		self.__set_lib(device=device, precision=precision)
		self.__get_version()

	def reset_device(self):
		if self.device_reset_allowed:
			for lib in self.__active_libs:
				if isinstance(lib, ct.CDLL):
					if 'ok_device_reset' in lib.__dict__:
						if lib.ok_device_reset():
							raise RuntimeError('device reset failed')
						return 0
			raise RuntimeError('device reset not possible: '
							   'no libraries loaded')
		else:
			raise RuntimeError('device reset not allowed: '
							   'C objects allocated')