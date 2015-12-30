from ctypes import CDLL, c_int, c_uint, c_size_t, c_void_p
from subprocess import check_output
from os import path, uname, getenv
from numpy import float32
from site import getsitepackages

class ProxLibs(object):
	def __init__(self):
		self.libs = {}
		local_c_build = path.abspath(path.join(path.dirname(__file__),
			'..', '..', '..', 'build'))
		search_results = ""
		use_local = getenv('OPTKIT_USE_LOCALLIBS', 0)

		# NB: no windows support
		ext = "dylib" if uname()[0] == "Darwin" else "so"
		for device in ['gpu', 'cpu']:
			for precision in ['32', '64']:
				lib_tag = '{}{}'.format(device, precision)
				lib_name = 'libprox_{}{}.{}'.format(device, precision, ext)
				lib_path = getsitepackages()[0]
				if not use_local and path.exists(path.join(lib_path, lib_name)):
					lib_path = path.join(lib_path, lib_name)
				else:
					lib_path = path.join(local_c_build, lib_name)

				try:
					lib = CDLL(lib_path)
					self.libs[lib_tag]=lib
				except (OSError, IndexError):
					search_results += str("library {} not found at {}.\n".format(
						lib_name, lib_path))
					self.libs[lib_tag]=None

		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, lowtypes, GPU=False):
		device = 'gpu' if GPU else 'cpu'
		precision = '32' if lowtypes.FLOAT_CAST == float32 else '64'
		lib_key = '{}{}'.format(device, precision)

		if self.libs[lib_key] is not None:
			lib = self.libs[lib_key]
			ok_float = lowtypes.ok_float
			ok_float_p = lowtypes.ok_float_p
			vector_p = lowtypes.vector_p
			function_p = lowtypes.function_p
			function_vector_p = lowtypes.function_vector_p

			# Function Vector 
			# ------
			## arguments
			lib.function_vector_alloc.argtypes=[function_vector_p, c_size_t]
			lib.function_vector_calloc.argtypes=[function_vector_p, c_size_t]
			lib.function_vector_free.argtypes=[function_vector_p]
			lib.function_vector_memcpy_va.argtypes=[function_vector_p, function_p]
			lib.function_vector_memcpy_av.argtypes=[function_p, function_vector_p]			
			lib.function_vector_print.argtypes=[function_vector_p]

			## return values
			lib.function_vector_alloc.restype=None
			lib.function_vector_calloc.restype=None
			lib.function_vector_free.restype=None
			lib.function_vector_memcpy_va.restype = None
			lib.function_vector_memcpy_av.restype = None			
			lib.function_vector_print.restype=None

			# Prox & Function evaluation
			# --------------------------
			## arguments
			lib.ProxEvalVector.argtypes=[function_vector_p, ok_float, vector_p, vector_p]
			lib.FuncEvalVector.argtypes=[function_vector_p, vector_p]

			## return values
			lib.ProxEvalVector.restype=None
			lib.FuncEvalVector.restype=ok_float

			return lib