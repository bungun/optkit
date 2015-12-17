from ctypes import CDLL, c_int, c_uint, c_size_t, c_void_p
from subprocess import check_output
from os import path, uname
from numpy import float32

class ProxLibs(object):
	def __init__(self):
		self.libs = {}
		local_c_build = path.abspath(path.join(path.dirname(__file__),
			'..', '..', '..', 'build'))
		search_results = ""


		# NB: no windows support
		ext = "dylib" if uname()[0] == "Darwin" else "so"
		for device in ['gpu', 'cpu']:
			for precision in ['32', '64']:
				lib_tag = '{}{}'.format(device, precision)
				lib_name = 'libprox_{}{}.{}'.format(device, precision, ext)
				lib_path = check_output(['locate', path.join('packages', lib_name)])
				if lib_path == '':
					lib_path = path.join(local_c_build, lib_name)
				elif lib_path[-1]=='\n': 
					lib_path=lib_path[:-1]
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
			c_uint_p = lowtypes.c_uint_p
			vector_p = lowtypes.vector_p
			function_p = lowtypes.function_p
			function_vector_p = lowtypes.function_vector_p

			# Function Vector 
			# ------
			## arguments
			lib.function_vector_alloc.argtypes=[function_vector_p, c_size_t]
			lib.function_vector_calloc.argtypes=[function_vector_p, c_size_t]
			lib.function_vector_free.argtypes=[function_vector_p]
			lib.function_vector_from_multiarray.argtypes=[function_vector_p, c_uint_p,
															  ok_float_p, ok_float_p, 
															  ok_float_p, ok_float_p, 
															  ok_float_p, c_size_t]
			lib.function_vector_memcpy_vmulti.argtypes=[function_vector_p, c_uint_p, 
															ok_float_p, ok_float_p, 
															ok_float_p, ok_float_p, 
															ok_float_p]
			lib.function_vector_print.argtypes=[function_vector_p]


			## return values
			lib.function_vector_alloc.restype=None
			lib.function_vector_calloc.restype=None
			lib.function_vector_free.restype=None
			lib.function_vector_from_multiarray.restype=None
			lib.function_vector_memcpy_vmulti.restype=None
			lib.function_vector_print.restype=None

			# Prox & Function
			# ---------------
			## arguments
			lib.ProxEvalVector.argtypes=[function_vector_p, ok_float, vector_p, vector_p]
			lib.FuncEvalVector.argtypes=[function_vector_p, vector_p]

			## return values
			lib.ProxEvalVector.restype=None
			lib.FuncEvalVector.restype=ok_float

			return lib