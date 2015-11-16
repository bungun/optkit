from optkit.defs import GPU_FLAG, FLOAT_CAST
from optkit.types import ok_function_enums as fcn_enums
from optkit.types.lowlevel import ok_float
from optkit.utils import make_cfunctionvector, release_cfunctionvector
from optkit.pyutils import istypedtuple
from ctypes import c_uint
from numpy import ones, zeros, ndarray


class FunctionVector(object):
	def __init__(self, *f, **params):
		# h = fcn_enums.safe_enum(params['h']) if 'h' in params else fcn_enums.Zero
		# a = float(params['a']) if 'a' in params else 1.
		# b = float(params['b']) if 'b' in params else 0.
		# c = max(float(params['c']),0.) if 'c' in params else 1.
		# d = float(params['d']) if 'd' in params else 0.
		# e = max(float(params['e']),0.) if 'e' in params else 0.



		valid = istypedtuple(f,1,int)
		print f[0]
		print valid
		# if len(f) > 0:
		# 	valid_local = True
		# 	for i in xrange(len(f)):
		# 		valid |= isinstance(f[i], (int, float, ndarray))
		# 		valid_local &= len(f[i].shape==1
		# 		if i == 0:
		# 			valid_local &= f[i].dtype==int
		# 			valid |= valid_local
		# 			valid |= isinstance(f[i],int)
		# 		else:
		# 			valid_local &= f[i].dtype==ok_float
		# 		valid |= valid_local
		# 		valid |= 



		# if len(f)==1:
		# 	if isinstance(f[0],ndarray):
		# 		valid |= len(f[0].shape)==1
		# elif len(f)==2:
		# 	if isinstance(x[0],ndarray) and isinstance(f[1],ok_function_vector):
		# 		valid |= ( len(f[0].shape)==1 and \
		# 					len(f[0])==f[1].size)			


		if not valid:
			print ("optkit.FunctionVector must be initialized with:\n"
					"-one `int`")
		
					# "-one `int` OR\n" 
		# 			"-one 1-dimensional `numpy.ndarray`, OR\n"
		# 			"-one 1-dimensional `numpy.ndarray` and"
		# 			" one `optkit.types.lowlevel.ok_vector` with"
		# 			" compatible dimensions)")
			self.on_gpu = None
			self.c = None
			self.h_ = None
			self.a_ = None
			self.b_ = None
			self.c_ = None
			self.d_ = None
			self.e_ = None
			self.size = None
			return

		self.h_ = ones(f[0], dtype=c_uint)
		self.a_ = ones(f[0], dtype=ok_float)
		self.b_ = zeros(f[0], dtype=ok_float)
		self.c_ = ones(f[0], dtype=ok_float)
		self.d_ = zeros(f[0], dtype=ok_float)
		self.e_ = zeros(f[0], dtype=ok_float)
		self.size = f[0];


		self.on_gpu = GPU_FLAG
		self.c = make_cfunctionvector(self.size)


	def __del__(self):
		if self.on_gpu: release_cfunctionvector(self.c)











