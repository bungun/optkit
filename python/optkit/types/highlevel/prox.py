from optkit.defs import GPU_FLAG, FLOAT_CAST
from optkit.types import ok_function_enums as fcn_enums
from optkit.types.lowlevel import function_dt
from optkit.utils import make_cfunctionvector, release_cfunctionvector
from optkit.pyutils import istypedtuple
from numpy import zeros, ndarray


class Function(object):
	pass
# 	def __init__(self, **params):
# 		self.h = fcn_enums.safe_enum(params['h']) if 'h' in params \
# 					else fcn_enums.Zero
# 		self.a = float(params['a']) if 'a' in params else 1.
# 		self.b = float(params['b']) if 'b' in params else 0.
# 		self.c = max(float(params['c']),0.) if 'c' in params else 1.
# 		self.d = float(params['d']) if 'd' in params else 0.
# 		self.e = max(float(params['e']),0.) if 'e' in params else 0.



class FunctionVector(object):
	def __init__(self, *f, **params):
		h = fcn_enums.safe_enum(params['h']) if 'h' in params else fcn_enums.Zero
		a = float(params['a']) if 'a' in params else 1.
		b = float(params['b']) if 'b' in params else 0.
		c = max(float(params['c']),0.) if 'c' in params else 1.
		d = float(params['d']) if 'd' in params else 0.
		e = max(float(params['e']),0.) if 'e' in params else 0.

		valid = istypedtuple(f,1,int)
		if len(f)==1:
			if isinstance(f[0],ndarray):
				valid |= len(f[0].shape)==1
		elif len(f)==2:
			if isinstance(x[0],ndarray) and isinstance(f[1],ok_function_vector):
				valid |= ( len(f[0].shape)==1 and \
							len(f[0])==f[1].size)			


		if not valid:
			data = None
			print ("optkit.FunctionVector must be initialized with:\n"
					"-one `int` OR\n" 
					"-one 1-dimensional `numpy.ndarray`, OR\n"
					"-one 1-dimensional `numpy.ndarray` and"
					" one `optkit.types.lowlevel.ok_vector` with"
					" compatible dimensions)")
			self.on_gpu = None
			self.sync_required = None
			self.py = None
			self.c = None
			self.size = None
			return



		self.on_gpu = GPU_FLAG
		if len(f)==1:
			if istypedtuple(f,1,int):
				data = zeros(f,dtype=function_dt)				
			else:
				# TODO: what is correct version of this?
				data = function_dt(f[0])


			self.py = data
			self.c = make_cfunctionvector(self.py, copy_data = GPU_FLAG)
			self.size = self.py.size
		else:
			self.py=f[0]
			self.c=f[1]
			self.size=f[1].size



	def __del__(self):
		if self.on_gpu: release_cfunctionvector(self.c)











