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
		function_vector_memcpy_va = backend.prox.function_vector_memcpy_va
		function_vector_memcpy_av = backend.prox.function_vector_memcpy_av
		ndarray_pointer = backend.lowtypes.ndarray_pointer

		class FunctionVector(object):
			def __init__(self, n, **params):
				backend.__LIBGUARD_ON__ = True
				if not isinstance(n,int):
					raise ValueError("optkit.FunctionVector must be initialized "
						"with:\n -one `int`")

				self.on_gpu = ON_GPU
				self.size = n;			
				self.c = make_cfunctionvector(self.size)
				self.py = zeros(n, dtype=backend.lowtypes.function)
				for i in xrange(n):	
					self.py[i]= backend.lowtypes.function(0, 1, 0, 1, 0, 0)
				self.set(**params)
				if 'f' in params:
					self.copy_from(params['f'])

			def copy_from(self, fv):
				if not isinstance(fv, FunctionVector):
					raise TypeError("FunctionVector copy() requires "
						"FunctionVector input")
				if not fv.size==self.size:
					raise ValueError("Incompatible dimensions")
				self.py[:] = fv.py[:]
				self.push()
 

			def tolist(self):
				return [backend.lowtypes.function(*self.py[i]) for i in xrange(self.size)]

			def set(self, **params):
				start = int(params['start']) if 'start' in params else 0
				end = int(params['end']) if 'end' in params else self.size
				range_length = len(self.py[start:end])
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



				objectives = self.tolist()

				#TODO: support complex slicing


				if 'h' in params:
					if isinstance(params['h'],(int, str)):
						for i in xrange(start, end):
							objectives[i].h = fcn_enums.safe_enum(params['h'])
					elif isinstance(params['h'],(list, ndarray)):
						for i in xrange(start, end):
							objectives[i].h = fcn_enums.safe_enum(params['h'][i])

				if 'a' in params:
					if isinstance(params['a'],(int, float)):
						for i in xrange(start, end):
							objectives[i].a = params['a']
					elif isinstance(params['a'],(list, ndarray)):
						for i in xrange(start, end):
							objectives[i].a = params['a'][i - start]

				if 'b' in params:
					if isinstance(params['b'],(int, float)):
						for i in xrange(start, end):
							objectives[i].b = params['b']
					elif isinstance(params['b'],(list, ndarray)):
						for i in xrange(start, end):
							objectives[i].b = params['b'][i - start]

				if 'c' in params:
					if isinstance(params['c'],(int,float)):
						for i in xrange(start, end):
							objectives[i].c = max(params['c'], 0)
					elif isinstance(params['c'],(list, ndarray)):
						for i in xrange(start, end):
							objectives[i].c = max(params['c'][i - start], 0)

				if 'd' in params:
					if isinstance(params['d'],(int, float)):
						for i in xrange(start, end):
							objectives[i].d = params['d']
					elif isinstance(params['d'],(list, ndarray)):
						for i in xrange(start, end):
							objectives[i].d = params['d'][i - start]

				if 'e' in params:
					if isinstance(params['e'],(int,float)):
						for i in xrange(start, end):
							objectives[i].e = max(params['e'], 0)
					elif isinstance(params['e'],(list, ndarray)):
						for i in xrange(start, end):
							objectives[i].e = max(params['e'][i - start], 0)

				for i in xrange(self.size):
					self.py[i] = objectives[i]
				self.push()
				
			def push(self):
				function_vector_memcpy_va(self.c, ndarray_pointer(self.py))	

			def pull(self):
				function_vector_memcpy_av(ndarray_pointer(self.py), self.c)	


			def __str__(self):
				self.pull()
				obj = self.tolist()
				h_ = zeros(self.size, dtype=int)
				a_ = zeros(self.size)
				b_ = zeros(self.size)
				c_ = zeros(self.size)
				d_ = zeros(self.size)
				e_ = zeros(self.size)

				for i in xrange(self.size):
					h_[i] = obj[i].h
					a_[i] = obj[i].a
					b_[i] = obj[i].b
					c_[i] = obj[i].c
					d_[i] = obj[i].d
					e_[i] = obj[i].e

				return str("size: {}\nc pointer: {}\non GPU?: {}\n"
					"h: {}\na: {}\nb: {}\nc: {}\nd: {}\ne: {}".format(
					self.size, self.c, self.on_gpu,
					h_, a_, b_, c_, d_, e_))

			def __del__(self):
				if self.on_gpu: release_cfunctionvector(self.c)

			def isvalid(self):
				for item in ['c','py','size','on_gpu']:
					assert self.__dict__.has_key(item)
					assert self.__dict__[item] is not None
				assert isinstance(self.py, ndarray)
				assert len(self.py.shape) == 1
				assert self.py.size == self.size
				assert self.size == self.c.size	
				return True	

		self.FunctionVector = FunctionVector