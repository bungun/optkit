from optkit.utils import istypedtuple
from numpy import zeros, ndarray

class HighLevelLinsysTypes(object):
	def __init__(self, backend):
		backend = backend
		ON_GPU = backend.device == 'gpu'
		lowtypes = backend.lowtypes
		make_cvector = backend.make_cvector
		make_cmatrix = backend.make_cmatrix
		release_cvector = backend.release_cvector
		release_cmatrix = backend.release_cmatrix

		class Vector(object):
			def __init__(self, *x, **flags):
				backend.__LIBGUARD_ON__ = True

				valid = istypedtuple(x, 1, int)
				if len(x)==1:
					if isinstance(x[0], ndarray):
						valid |= len(x[0].shape) == 1
				elif len(x)==2:
					if isinstance(x[0], ndarray) and isinstance(x[1], lowtypes.vector):
						valid |= ( len(x[0].shape) == 1 and \
									len(x[0]) == x[1].size)			


				if not valid:
					data = None
					print ("optkit.Vector must be initialized with:\n"
							"-one `int` OR\n" 
							"-one 1-dimensional `numpy.ndarray`, OR\n"
							"-one 1-dimensional `numpy.ndarray` and"
							" one `optkit.types.lowlevel.ok_vector` with"
							" compatible dimensions)")
					self.is_view = None
					self.on_gpu = None
					self.sync_required = None
					self.py = None
					self.c = None
					self.size = None
					return


				self.is_view = 'is_view' in flags
				self.on_gpu = ON_GPU
				self.sync_required = ON_GPU 

				if len(x) == 1:
					if istypedtuple(x, 1, int):
						data = zeros(x, dtype=lowtypes.FLOAT_CAST)				
					else:
						data = zeros(len(x[0]), dtype=lowtypes.FLOAT_CAST)
						data[:] = x[0][:]

					self.py = data
					self.c = make_cvector(self.py, copy_data = ON_GPU)
					self.size = self.py.size
				else:
					# --------------------------------------------- #
					# python and C arrays provided to constructor, 	#
					# used for constructing views					#
					# --------------------------------------------- #
					self.py = x[0]
					self.c = x[1]
					self.size = x[1].size
					self.sync_required |= 'sync_required' in flags

			def __str__(self):
				return str("PY: {},\nC: {},\nSIZE: {}\n"
							  "on GPU? {}\nsync required? {}\n".format(
							  str(self.py), str(self.c), self.size, 
							  	self.on_gpu, self.sync_required))

			def __del__(self):
				if self.is_view: return
				if self.on_gpu: release_cvector(self.c)

			def isvalid(self):
				for item in ['on_gpu','sync_required','is_view','size','c','py']:
					assert self.__dict__.has_key(item)
					assert self.__dict__[item] is not None
				if self.on_gpu:
					assert self.sync_required
				assert isinstance(self.py, ndarray)
				assert len(self.py.shape) == 1
				assert self.py.size == self.size
				assert isinstance(self.c, lowtypes.vector)
				assert self.c.size == self.size
				assert self.c.data is not None
				return True

		self.Vector = Vector

		class Matrix(object):

			def __init__(self, *A, **flags):
				backend.__LIBGUARD_ON__ = True

				# args are (int, int)
				valid = istypedtuple(A,2,int)
				if not valid:
					if len(A)==1:
						# args are (ndarray)
						if isinstance(A[0],ndarray):
							# ndarray is matrix
							valid |= len(A[0].shape)==2
					elif len(A)==2:
						# args are (ndarray, ok_matrix)
						if isinstance(A[0],ndarray) and isinstance(A[1],lowtypes.matrix):
							# ndarray is matrix
							if len(A[0].shape)==2:
								# ndarray and ok-matrix compatibly sized
								valid |= A[0].shape[0] == A[1].size1 and \
										 A[0].shape[1] == A[1].size2			
					

				if not valid:
					print ("optkit.Matrix must be initialized with\n"
							"-two `int` arguments OR\n"
							"-one 2-dimensional `numpy.ndarray` OR\n"
							"-one 2-dimensional `numpy.ndarray` and"
							" one `optkit.types.lowlevel.ok_matrix`"
							" of compatible sizes.")
					self.is_view = None
					self.on_gpu = None
					self.sync_required = None
					self.py = None
					self.c = None
					self.size1 = None
					self.size2 = None
					self.shape = None
					self.skinny = None
					self.mindim = None
					return

				order = 'F' if backend.layout == 'col' else 'C'

				self.is_view = 'is_view' in flags
				self.on_gpu = ON_GPU
				self.sync_required = ON_GPU
				if len(A)==1 or istypedtuple(A,2,int):
					if len(A)==1:
						data = zeros(A[0].shape,dtype=lowtypes.FLOAT_CAST, 
							order=order)
						data[:]=A[0][:]
					else:
						data = zeros(A,dtype=lowtypes.FLOAT_CAST,
							order=order)

					self.py = data
					self.c = make_cmatrix(self.py, copy_data = ON_GPU)
					self.size1 = self.py.shape[0]
					self.size2 = self.py.shape[1]
				else:

					# --------------------------------------------- #
					# python and C arrays provided to constructor, 	#
					# used for constructing views					#
					# --------------------------------------------- #
					self.py = A[0]
					self.c = A[1]
					self.size1 = A[1].size1
					self.size2 = A[1].size2
				self.skinny = self.size1 >= self.size2
				self.square = self.size1 == self.size2
				self.mindim = min(self.size1, self.size2)
				self.shape = (self.size1, self.size2)

			def __str__(self):
				return str("PY: {},\nC: {},\nSIZE: ({},{}),\n"
							  "on GPU? {}\nsync required? {}\n".format(
							  str(self.py), str(self.c), 
							  self.size1,self.size2, 
							  self.on_gpu, self.sync_required))


			def __del__(self):
				if self.is_view: return
				if self.on_gpu: release_cmatrix(self.c)


			def isvalid(self):
				for item in ['on_gpu','sync_required','shape','size1',
							'size2','skinny','mindim','c','py']:
					assert self.__dict__.has_key(item)
					assert self.__dict__[item] is not None
				if self.on_gpu:
					assert self.sync_required
				assert self.shape == (self.size1, self.size2)
				assert self.skinny == (self.size1 >= self.size2)
				assert self.mindim == min(self.size1, self.size2)
				assert isinstance(self.py, ndarray)
				assert self.py.shape == self.shape
				assert isinstance(self.c, lowtypes.matrix)
				assert self.c.size1 == self.size1
				assert self.c.size2 == self.size2
				assert self.c.rowmajor
				assert self.c.data is not None
				return True

		self.Matrix = Matrix


class Range(object):
	def __init__(self, ubound, idx1, idx2=None):
		self.idx1 = int(min(max(idx1, 0), ubound - 1))
		if idx2 is None:
			self.idx2 = self.idx1 + 1
		else:
			self.idx2 = max(idx1 + 1, int(min(idx2, ubound)))
		self.elements = self.idx2 - self.idx1

	def __str__(self):
		return "index 1: {}\nindex2: {}\n length: {}".format(
			self.idx1, self.idx2, self.elements)