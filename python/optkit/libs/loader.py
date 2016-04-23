from os import path, uname, getenv
from site import getsitepackages
from ctypes import CDLL
from optkit.libs.enums import OKEnums

def retrieve_libs(lib_prefix):
	libs = {}
	local_c_build = path.abspath(path.join(path.dirname(__file__),
		'..', '..', '..', 'build'))
	search_results = '\n'
	use_local = getenv('OPTKIT_USE_LOCALLIBS', 0)

	# NB: no windows support
	ext = "dylib" if uname()[0] == "Darwin" else "so"

	for device in ['gpu', 'cpu']:
		for precision in ['32', '64']:
			lib_tag = '{}{}'.format(device, precision)
			lib_name = '{}{}{}.{}'.format(lib_prefix, device, precision, ext)
			lib_path = path.join(getsitepackages()[0], '_optkit_libs',
								 lib_name)
			if use_local or not path.exists(lib_path):
				lib_path = path.join(local_c_build, lib_name)

			if path.exists(lib_path):
				libs[lib_tag] = CDLL(lib_path)
				libs[lib_tag].INITIALIZED = False
			else:
				msg = 'library {} not found at {}.\n'.format(lib_name, lib_path)
				search_results += msg
				libs[lib_tag] = None

	return libs, search_results

class OptkitLibs(object):
	def __init__(self, lib_prefix):
		self.libs, search_results = retrieve_libs(lib_prefix)
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

		self.attach_calls = []

	def get(self, single_precision=False, gpu=False):
		device = 'gpu' if gpu else 'cpu'
		precision = '32' if single_precision else '64'
		lib_key = '{}{}'.format(device, precision)
		if lib_key not in self.libs:
			return None
		elif self.libs[lib_key] is None:
			return None

		lib = self.libs[lib_key]
		if lib.INITIALIZED:
			return lib
		else:
			lib.enums = OKEnums()

			for attach_call in self.attach_calls:
				attach_call(lib, single_precision)

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib