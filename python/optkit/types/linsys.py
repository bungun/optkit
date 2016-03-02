from numpy import zeros, ndarray
from scipy.sparse import csr_matrix, csc_matrix
from optkit.utils import istypedtuple

class LinsysTypes(object):
	def __init__(self, backend):
		ON_GPU = backend.device == 'gpu'

		class Vector(object):
			def __init__(self, *x, **flags):
				if not backend.linalg_contexts_exist:
					backend.make_linalg_contexts()
				backend.increment_cobject_count()

				valid = istypedtuple(x, 1, int)
				if len(x)==1:
					if isinstance(x[0], ndarray):
						valid |= len(x[0].shape) == 1
				elif len(x)==2:
					if isinstance(x[0], ndarray) and isinstance(x[1],
						backend.dense.vector):
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
						data = zeros(x, dtype=backend.dense.pyfloat)
					else:
						data = zeros(len(x[0]), dtype=backend.dense.pyfloat)
						data[:] = x[0][:]

					self.py = data
					self.c = backend.make_cvector(self.py, copy_data = ON_GPU)
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
				backend.decrement_cobject_count()
				if self.is_view: return
				if self.on_gpu: backend.release_cvector(self.c)

			def isvalid(self):
				for item in ['on_gpu', 'sync_required', 'is_view', 'size', 'c',
					'py']:
					assert self.__dict__.has_key(item)
					assert self.__dict__[item] is not None
				if self.on_gpu:
					assert self.sync_required
				assert isinstance(self.py, ndarray)
				assert len(self.py.shape) == 1
				assert self.py.size == self.size
				assert isinstance(self.c, backend.dense.vector)
				assert self.c.size == self.size
				assert self.c.data is not None
				return True

		self.Vector = Vector

		class Matrix(object):
			def __init__(self, *A, **flags):
				if not backend.linalg_contexts_exist:
					backend.make_linalg_contexts()
				backend.increment_cobject_count()

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
						if isinstance(A[0],ndarray) and isinstance(A[1],
							backend.dense.matrix):
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
						data = zeros(A[0].shape,dtype=backend.dense.pyfloat,
							order=order)
						data[:]=A[0][:]
					else:
						data = zeros(A,dtype=backend.dense.pyfloat,
							order=order)

					self.py = data
					self.c = backend.make_cmatrix(self.py, copy_data = ON_GPU)
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
				backend.decrement_cobject_count()
				if self.is_view: return
				if self.on_gpu: backend.release_cmatrix(self.c)


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
				assert isinstance(self.c, backend.matrix)
				assert self.c.size1 == self.size1
				assert self.c.size2 == self.size2
				assert self.c.order
				assert self.c.data is not None
				return True

		self.Matrix = Matrix

		class SparseMatrix(object):

			def __init__(self, *A, **flags):
				if not backend.linalg_contexts_exist:
					backend.make_linalg_contexts()
				backend.increment_cobject_count()

				# args are (int, int, int)
				valid = istypedtuple(A, 3, int)
				if not valid:
					if len(A)==1:
						# args are (ndarray)
						if isinstance(A[0], (csr_matrix, csc_matrix)):
							# ndarray is matrix
							valid |= len(A[0].shape)==2
					elif len(A)==2:
						# args are (scipy.sparse matrix, ok_sparse_matrix)
						if isinstance(A[0], (csr_matrix, csc_matrix)) and \
							isinstance(A[1], backend.sparse.sparse_matrix):
							# ndarray and ok-matrix compatibly sized
							valid |= A[0].shape[0] == A[1].size1 and \
									 A[0].shape[1] == A[1].size2

				if not valid:
					print ("optkit.SparseMatrix must be initialized with\n"
						"-three `int` arguments OR\n"
						"-one `scipy.sparse.csc_matrix` (or csr_matrix) OR\n"
						"-one `scipy.sparse.csc_matrix` (or csr_matrix) and"
						" one `optkit.types.lowlevel.ok_matrix`"
						" of compatible sizes.")
					self.on_gpu = None
					self.sync_required = None
					self.py = None
					self.c = None
					self.size1 = None
					self.size2 = None
					self.nnz = None
					self.shape = None
					self.skinny = None
					self.mindim = None
					self.ptrlen = None
					return

				spmat_constructor = csc_matrix if backend.layout == 'col' else \
					csr_matrix

				self.on_gpu = ON_GPU
				self.sync_required = ON_GPU
				if istypedtuple(A, 3, int) or \
					istypedtuple(A, 1, csc_matrix) or \
					istypedtuple(A, 1, csr_matrix):

					if len(A)==1:
						data = spmat_constructor(A[0].astype(
							backend.dense.pyfloat))
					else:
						(m, n, nnz) = A
						if nnz > m * n:
							print str("# nonzeros nnz exceeds "
								"matrix dimensions m * n, using  "
								"nnz = m * n instead")
							nnz = m * n

						shape = (m, n)
						ptrlen = n + 1 if backend.layout == 'col' else m + 1
						ptr = zeros(ptrlen, dtype=int)
						ind = zeros(nnz, dtype=int)


						val = zeros(nnz, dtype=lowtypes.backend.dense.pyfloat)
						ptr_idx = 0
						# get size of complementary (non-compressed) dimension
						modbase = n + m + 1 - ptrlen

						# create a default sequentially filled initialization of
						# the sparse matrix so that the csc_matrix() or
						# csr_matrix() constructor allocates & returns
						# something with nnz nonzeros
						for i in xrange(nnz):
							ind[i] = i % modbase
							ptr_idx = i / modbase
							ptr[ptr_idx + 1] = i + 1
						if ptr_idx < ptrlen:
							ptr[ptr_idx + 1:] = nnz

						data = spmat_constructor((val, ind, ptr), shape=shape)

					self.py = data
					self.c = backend.make_csparsematrix(self.py)
					self.size1 = self.py.shape[0]
					self.size2 = self.py.shape[1]
					self.nnz = self.py.nnz
					self.ptrlen = data.shape[1] + 1 if backend.layout == 'col' \
						else data.shape[1] + 1
				else:
					# --------------------------------------------- #
					# python and C arrays provided to constructor, 	#
					# used for constructing views					#
					# --------------------------------------------- #
					self.py = A[0]
					self.c = A[1]
					self.size1 = A[1].size1
					self.size2 = A[1].size2
					self.nnz = A[1].nnz
					self.ptrlen = A[1].ptrlen
				self.skinny = self.size1 >= self.size2
				self.square = self.size1 == self.size2
				self.mindim = min(self.size1, self.size2)
				self.shape = (self.size1, self.size2)

			def __str__(self):
				return str("PY: {},\nC: {},\nSIZE: ({},{}), NNZ: {}\n"
							  "on GPU? {}\nsync required? {}\n".format(
							  str(self.py), str(self.c),
							  self.size1, self.size2, self.nnz,
							  self.on_gpu, self.sync_required))


			def __del__(self):
				backend.decrement_cobject_count()
				if self.on_gpu: backend.release_csparsematrix(self.c)


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
				assert isinstance(self.py, (csr_matrix, csc_matrix))
				assert self.py.shape == self.shape
				assert isinstance(self.c, lowtypes.sparse_matrix)
				assert self.c.size1 == self.size1
				assert self.c.size2 == self.size2
				assert self.c.nnz == self.nnz
				assert self.c.val is not None
				assert self.c.ind is not None
				assert self.c.ptr is not None
				return True

		self.SparseMatrix = SparseMatrix

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