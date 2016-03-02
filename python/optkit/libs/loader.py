from os import path, uname, getenv
from site import getsitepackages
from ctypes import CDLL

def retrieve_libs(lib_prefix):
	libs = {}
	local_c_build = path.abspath(path.join(path.dirname(__file__),
		'..', '..', '..', 'build'))
	search_results = ''
	use_local = getenv('OPTKIT_USE_LOCALLIBS', 0)

	# NB: no windows support
	ext = "dylib" if uname()[0] == "Darwin" else "so"

	for device in ['gpu', 'cpu']:
		for precision in ['32', '64']:
			lib_tag = '{}{}'.format(device, precision)
			lib_name = '{}{}{}.{}'.format(lib_prefix, device, precision, ext)
			lib_path = path.join(getsitepackages()[0], lib_name)
			if not use_local or not path.exists(lib_path):
				lib_path = path.join(local_c_build, lib_name)

			if path.exists(lib_path):
				libs[lib_tag] = CDLL(lib_path)
				libs[lib_tag].INITIALIZED = False
			else:
				msg = 'library {} not found at {}.\n'.format(lib_name, lib_path)
				search_results.join(msg)
				libs[lib_tag] = None

	return libs, search_results

def validate_lib(lib, libname, expected_call, requestor, single_precision, gpu):
		if not isinstance(lib, CDLL):
			TypeError('argument "{}" must be of type {}'.format(libname, CDLL))
		elif expected_call not in dir(lib):
			ValueError('argument "{}" does not appear to correspond'
				'to the expected library ("{} is missing)'.format(libname,
					expected_call))
		elif 'FLOAT' not in dir(lib):
			ValueError('argument "{}" is missing specification field "FLOAT" '
				'and is consquently unusable'.format(libname))
		elif 'GPU' not in dir(lib):
			ValueError('argument "{}" is missing specification field "GPU" '
				'and is consquently unusable'.format(libname))
		elif lib.FLOAT != single_precision:
			ValueError('floating point precision of library given by '
				'argument "denselib" specifies single_precision = {} '
				'current {}.get() call specifes single_precision = {}'.format(
					lib.FLOAT, requestor, single_precision))
		elif lib.GPU != gpu:
			ValueError('floating point precision of library given by '
				'argument "denselib" specifies gpu = {} '
				'current {}.get() call specifes gpu = {}'.format(lib.GPU,
					requestor, gpu))