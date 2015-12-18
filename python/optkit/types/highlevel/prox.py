from optkit.types import ok_function_enums as fcn_enums
from optkit.utils import UtilMakeCFunctionVector, UtilReleaseCFunctionVector
from ctypes import c_uint
from numpy import ones, zeros, ndarray


class HighLevelProxTypes(object):
	def __init__(self, backend):
		backend = backend
		ON_GPU = backend.device == 'gpu'
		make_cfunctionvector = backend.make_cfunctionvector
		release_cfunctionvector = backend.release_cfunctionvector

		class FunctionVector(object):
			def __init__(self, n, **params):
				backend.__LIBGUARD_ON__ = True
				if not isinstance(n,int):
					raise ValueError("optkit.FunctionVector must be initialized "
						"with:\n -one `int`")
				
				self.h_ = ones(n, dtype=c_uint)
				self.a_ = ones(n, dtype=backend.lowtypes.ok_float)
				self.b_ = zeros(n, dtype=backend.lowtypes.ok_float)
				self.c_ = ones(n, dtype=backend.lowtypes.ok_float)
				self.d_ = zeros(n, dtype=backend.lowtypes.ok_float)
				self.e_ = zeros(n, dtype=backend.lowtypes.ok_float)
				self.size = n;
				self.set(**params)
				self.on_gpu = ON_GPU
				self.c = make_cfunctionvector(self.size)
				if 'f' in params:
					self.copy_from(params['f'])

			def copy_from(self, fv):
				if not isinstance(fv, FunctionVector):
					raise TypeError("FunctionVector copy() requires "
						"FunctionVector input")
				if not fv.size==self.size:
					raise ValueError("Incompatible dimensions")
				self.h_[:]=fv.h_[:]
				self.a_[:]=fv.a_[:]
				self.b_[:]=fv.b_[:]
				self.c_[:]=fv.c_[:]
				self.d_[:]=fv.d_[:]
				self.e_[:]=fv.e_[:]



			def set(self, **params):
				start = int(params['start']) if 'start' in params else None
				end = int(params['end']) if 'end' in params else None
				range_length = len(self.a_[start:end])
				if  range_length == 0: 
					raise ValueError('index range [{}:{}] results in length-0 array '
						'when python array slicing applied to a FunctionVector '
						' of length {}.'.format(start,end,self.size))
				for item in ['a', 'b', 'c', 'd', 'e', 'h']:
					if item in params:
						if isinstance(params[item],(list, ndarray)):
							if len(params[item]) != range_length:
								raise ValueError('keyword argument {} of type {} '
									'is incomptably sized with the requested '
									'FunctionVector slice [{}:{}]'.format(
									item, type(params(item), start, end)))


				#TODO: support complex slicing

				if 'h' in params:
					if isinstance(params['h'],(int,str)):
						self.h_[start:end]=fcn_enums.safe_enum(params['h'])
					elif isinstance(params['h'],(list, ndarray)):
						self.h_[start:end]=map(lambda v : fcn_enums.safe_enum(v), params['h'])
				if 'a' in params:
					if isinstance(params['a'],(int,float)):
						self.a_[start:end]=params['a']
					elif isinstance(params['a'],(list, ndarray)):
						self.a_[start:end]=params['a'][:]
				if 'b' in params:
					if isinstance(params['b'],(int,float)):
						self.b_[start:end]=params['b']
					elif isinstance(params['b'],(list, ndarray)):
						self.b_[start:end]=params['b'][:]
				if 'c' in params:
					if isinstance(params['c'],(int,float)):
						self.c_[start:end]=max(params['c'],0)
					elif isinstance(params['c'],(list, ndarray)):
						self.c_[start:end]=map(lambda v : max(v,0),params['c'])
				if 'd' in params:
					if isinstance(params['d'],(int,float)):
						self.d_[start:end]=params['d']
					elif isinstance(params['d'],(list, ndarray)):
						self.d_[start:end]=params['d'][:]
				if 'e' in params:
					if isinstance(params['e'],(int,float)):
						self.e_[start:end]=max(params['e'],0)
					elif isinstance(params['e'],(list, ndarray)):
						self.e_[start:end]=map(lambda v : max(v,0),params['e'][:])

			def __str__(self):
				return str("size: {}\nc pointer: {}\non GPU?: {}\n"
					"h: {}\na: {}\nb: {}\nc: {}\nd: {}\ne: {}".format(
					self.size, self.c, self.on_gpu,
					self.h_, self.a_, self.b_, self.c_, self.d_, self.e_))

			def __del__(self):
				if self.on_gpu: release_cfunctionvector(self.c)

			def isvalid(self):
				for item in ['c','h_','a_','b_','c_','d_','e_','size','on_gpu']:
					assert self.__dict__.has_key(item)
					assert self.__dict__[item] is not None
				for item in ['h_','a_','b_','c_','d_','e_']:
					assert isinstance(self.__dict__[item],ndarray)
					assert len(self.__dict__[item].shape) == 1
					assert self.__dict__[item].size == self.size 
				assert self.size == self.c.size	
				return True	

		self.FunctionVector=FunctionVector