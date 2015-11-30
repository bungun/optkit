from optkit.defs import GPU_FLAG, FLOAT_CAST
from optkit.types import ok_function_enums as fcn_enums
from optkit.types.lowlevel import ok_float
from optkit.utils import make_cfunctionvector, release_cfunctionvector,\
							istypedtuple
from ctypes import c_uint
from numpy import ones, zeros, ndarray


class FunctionVector(object):
	def __init__(self, n, **params):
		if not isinstance(n,int):
			raise ValueError("optkit.FunctionVector must be initialized "
				"with:\n -one `int`")
		
		self.h_ = ones(n, dtype=c_uint)
		self.a_ = ones(n, dtype=ok_float)
		self.b_ = zeros(n, dtype=ok_float)
		self.c_ = ones(n, dtype=ok_float)
		self.d_ = zeros(n, dtype=ok_float)
		self.e_ = zeros(n, dtype=ok_float)
		self.size = n;
		self.set(**params)
		self.on_gpu = GPU_FLAG
		self.c = make_cfunctionvector(self.size)

	def set(self, **params):
		if 'h' in params:
			if isinstance(params['h'],(int,str)):
				self.h_[:]=fcn_enums.safe_enum(params['h'])
			elif isinstance(params['h'],ndarray):
				self.h_[:]=map(lambda v : fcn_enums.safe_enum(v), params['h'])
		if 'a' in params:
			if isinstance(params['a'],(int,float)):
				self.a_[:]=params['a']
			elif isinstance(params['a'],ndarray):
				self.a_[:]=params['a'][:]
		if 'b' in params:
			if isinstance(params['b'],(int,float)):
				self.b_[:]=params['b']
			elif isinstance(params['b'],ndarray):
				self.b_[:]=params['b'][:]
		if 'c' in params:
			if isinstance(params['c'],(int,float)):
				self.c_[:]=max(params['c'],0)
			elif isinstance(params['c'],ndarray):
				self.c_[:]=map(lambda v : max(v,0),params['c'])
		if 'd' in params:
			if isinstance(params['d'],(int,float)):
				self.d_[:]=params['d']
			elif isinstance(params['d'],ndarray):
				self.d_[:]=params['d'][:]
		if 'e' in params:
			if isinstance(params['e'],(int,float)):
				self.e_[:]=max(params['e'],0)
			elif isinstance(params['e'],ndarray):
				self.e_[:]=map(lambda v : max(v,0),params['e'][:])

	def __str__(self):
		return str("size: {}\nc pointer: {}\non GPU?: {}\n"
			"h: {}\na: {}\nb: {}\nc: {}\nd: {}\ne: {}".format(
			self.size, self.c, self.on_gpu,
			self.h_, self.a_, self.b_, self.c_, self.d_, self.e_))

	def __del__(self):
		if self.on_gpu: release_cfunctionvector(self.c)











