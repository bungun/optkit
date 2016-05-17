from numpy import array, ndarray, zeros, ceil
from ctypes import c_size_t

def nearest_triple(factor):
	if not isinstance(factor, int):
		raise TypeError('argument "factor" must be of type {}'.format(int))
	index = 0
	triple = [1, 1, 1]
	curr_prod = 1
	last_prod = 0

	while curr_prod < factor:
		index = (index + 1) % 3
		triple[index] += 1
		last_prod = curr_prod
		curr_prod = triple[0] * triple[1] * triple[2]

	if abs(curr_prod - factor) <= abs(last_prod - factor):
		return tuple(triple)
	else:
		triple[index] -= 1
		return tuple(triple)

def regcluster(sample_dims, full_dims):
	valid = isinstance(sample_dims, tuple) and isinstance(full_dims, tuple)
	valid &= len(sample_dims) == 3
	valid &= len(full_dims) == 3
	valid &= all([isinstance(i, int) for i in sample_dims])
	valid &= all([isinstance(i, int) for i in full_dims])
	if not valid:
		raise TypeError('arguments "sample_dims" and "full_dims" must be '
						'length-3 integer tuples')

	nx, ny, nz = sample_dims
	xmax, ymax, zmax = full_dims

	x_blocks = int((xmax % nx)!=0) + xmax / nx
	y_blocks = int((ymax % ny)!=0) + ymax / ny
	z_blocks = int((zmax % nz)!=0) + zmax / nz

	xceil = nx * x_blocks
	yceil = ny * y_blocks
	zceil = nz * z_blocks

	grid = zeros((xceil, yceil, zceil), dtype=int)

	for k in xrange(zceil):
		for j in xrange(yceil):
			for i in xrange(xceil):
				grid[i, j, k] = (i / nx) * 1
				grid[i, j, k] += (j / ny) * x_blocks
				grid[i, j, k] += (k / nz) * y_blocks * x_blocks

	return grid[:xmax, :ymax, :zmax].reshape(xmax * ymax * zmax)

def geometric_cluster(dimx, dimy, dimz, downsampling_factor, perm_geom2working,
					  n_working, perm_working2sorted, block_sizes):

	geom = tuple(dimx, dimy, dimz)
	assignments = regcluster(nearest_triple(downsampling_factor),  geom)
	assignments = assignments[perm_geom2working.indices][:n_working]
	assignments = assignments[perm_working2sorted]

	entry_offset = 0

	assignment_blocks = []

	for b in xrange(block_sizes):
		size_b = block_sizes[b]
		assignments_b = UpsamplingVector(
				indices=assignments[entry_offset : entry_offset + size_b])
		assignments_b.rebase()
		assignments_b.clean_assignments()

		assignment_blocks.append(assignments_b)
		entry_offset += size_b

	return assignment_blocks

class UpsamplingVector(object):
	def __init__(self, size1=None, size2=None, indices=None):
		self.__size1 = None
		self.___size2 = None
		self.__indices = None
		self.__counts = None

		if indices is not None:
			self.indices = indices
		elif size1 is not None and size2 is not None:
			self.__size1 = int(size1)
			self.__size2 = int(size2)
			self.__indices = zeros(size1, dtype=int)
			self.__counts = zeros(size2, dtype=int)
		else:
			raise ValueError('initialization requires arguments "indices" or '
							 '"size1" AND "size2" to be provided')

	@property
	def size1(self):
		return self.__size1

	@property
	def size2(self):
		return self.__size2

	@property
	def indices(self):
		return self.__indices

	@indices.setter
	def indices(self, indices):
		if not isinstance(indices, ndarray):
			raise TypeError('argument "indices" must be of type {}'
							''.format(ndarray))

		self.__size1 = len(indices)
		self.__size2 = int(indices.max()) + 1
		self.__indices = indices.astype(int)
		self.__counts = zeros(self.__size2, dtype=int)

	@property
	def min_assignment(self):
		return self.__indices.min()

	@property
	def max_assignment(self):
		return self.__indices.max()

	def rebase_assignments(self, base=0):
		base = int(base)
		minidx = self.__indices.min()
		self.__indices += int(base - minidx)
		self.__size2 = self.max_assignment + 1

	def clean_assignments(self):
		""" eliminate non-contiguous assignments
		"""
		idx_order = self.indices.argsort()

		unit_incr = 0
		curr = self.min_assignment

		for idx in idx_order:
			if self.indices[idx] != curr:
				unit_incr += 1
				curr = self.indices[idx]

			self.indices[idx] = unit_incr

	def recalculate_size(self):
		self.__size2 = self.max_assignment + 1

	@property
	def counts(self):
		self.__counts *= 0
		for idx in self.__indices:
			self.__counts[idx] += 1
		return self.__counts

class ClusteringTypes(object):
	def __init__(self, backend):
		if backend.cluster is None:
			backend.load_lib('cluster')
		lib = backend.cluster

		class ClusteringSettings(object):
			DISTANCE_RTOL_DEFAULT = 2e-2
			REASSIGN_RTOL_DEFAULT = 1e-2
			MAXITER_DEFAULT = 500
			VERBOSE_DEFAULT = 1

			def __init__(self, m, distance_tol=2e-2, assignment_tol=1e-2,
						 maxiter=500, verbose=1):
				self.distance_tol = float(distance_tol)
				self.assignment_tol = int(ceil(assignment_tol * m))
				self.maxiter = int(maxiter)
				self.verbose = abs(int(verbose))

			@property
			def pointer(self):
				return lib.kmeans_settings(self.distance_tol, self.assignment_tol,
										   self.maxiter, self.verbose)

		self.ClusteringSettings = ClusteringSettings

		class ClusteringWork(object):
			def __init__(self, m, k, n):
				self.m = m
				self.k = k
				self.n = n
				self.m_max = m
				self.k_max = k
				self.n_max = n
				self.__kmeans_work = None
				self.__register_kmeans_work(lib, lib.kmeans_easy_init(m, k, n))

			def __del__(self):
				self.__unregister_kmeans_work()

			def __register_kmeans_work(self, lib, kmeans_work):
					backend.increment_cobject_count()
					self.__kmeans_work = kmeans_work

			def __unregister_kmeans_work(self):
				if self.pointer is None:
					return
				lib.kmeans_easy_finish(self.pointer)
				self.__kmeans_work = None
				backend.decrement_cobject_count()

			@property
			def pointer(self):
				return self.__kmeans_work

			def resize(self, m, k, n=None):
				n = self.n if n is None else n
				M = self.m_max
				K = self.k_max
				N = self.n_max
				if m > M or k > K or n > N:
					return ValueError('resize dimensions exceed maximum size of '
									  'm={}, k={}, n={} as determined by size at '
									  'initialization'.format(M, K, N))

				self.n = n
				self.k = k
				self.m = m
				return lib.kmeans_easy_resize(self.pointer, m, k, n)

		self.ClusteringWork = ClusteringWork

		class Clustering(object):

			def __init__(self):
				self.__kmeans_work = None
				self.__A = None
				self.__A_ptr = None
				self.__order_A = None
				self.__C = None
				self.__C_ptr = None
				self.__a2c = None
				self.__a2c_ptr = None
				self.__counts = None
				self.__counts_ptr = None

			def __del__(self):
				self.kmeans_work_finish()

			def __get_order(self, array_):
				if array_.flags.c_contiguous:
					return lib.enums.CblasRowMajor
				else:
					return lib.enums.CblasColMajor

			@property
			def kmeans_work(self):
				return self.__kmeans_work

			def kmeans_work_init(self, m, k, n):
				self.__kmeans_work = ClusteringWork(m, k, n)

			def kmeans_work_finish(self):
				if self.__kmeans_work is not None:
					self.__kmeans_work.__del__()
					self.__kmeans_work = None

			@property
			def A(self):
				return self.__A

			@A.setter
			def A(self, A):
				if not isinstance(A, ndarray) and len(A.shape) != 2:
					raise ValueError('argument "A" must be a 2-D {}'.format(
						ndarray))
				self.__A = A.astype(lib.pyfloat)
				self.__A_ptr = self.A.ctypes.data_as(lib.ok_float_p)
				self.__order_A = self.__get_order(self.A)

			@property
			def A_ptr(self):
				return self.__A_ptr

			@property
			def order_A(self):
				return self.__order_A

			@property
			def C(self):
				return self.__C

			@C.setter
			def C(self, C):
				if not isinstance(C, ndarray) or len(C.shape) != 2:
					raise ValueError('argument "C" must be a 2-D {}'.format(
						ndarray))

				self.__C = C.astype(lib.pyfloat)
				self.__C_ptr = self.C.ctypes.data_as(lib.ok_float_p)
				self.__order_C = self.__get_order(self.C)
				self.__counts = zeros(self.C.shape[0], dtype=lib.pyfloat)
				self.__counts_ptr = self.counts.ctypes.data_as(lib.ok_float_p)

			@property
			def C_ptr(self):
				return self.__C_ptr

			@property
			def order_C(self):
				return self.__order_C

			@property
			def a2c(self):
				return self.__a2c

			@a2c.setter
			def a2c(self, assignments):
				if isinstance(assignments, ndarray) and len(assignments.shape) == 1:
					self.__a2c = assignments.astype(c_size_t)
					self.__a2c_ptr = self.a2c.ctypes.data_as(lib.c_size_t_p)

			@property
			def a2c_ptr(self):
				return self.__a2c_ptr

			@property
			def counts(self):
				return self.__counts

			@property
			def counts_ptr(self):
				return self.__counts_ptr

			@property
			def io(self):
				if self.A is None or self.C is None or self.a2c is None:
					raise RuntimeError('fields "A", "C" and "a2c" need to be '
									   'initialized to generate a {} object'
									   ''.format(lib.kmeans_io))
				stride_counts = 1
				stride_a2c = 1
				return lib.kmeans_io(self.A_ptr, self.C_ptr, self.counts_ptr,
									 self.a2c_ptr, self.order_A, self.order_C,
									 stride_a2c, stride_counts)

			def kmeans_inplace(self, A, C, assignments, settings=None):
				# TODO: check dimension compatibility
				self.A = A
				self.C = C
				self.a2c = assignments
				m = A.shape[0]
				n = A.shape[1]
				k = C.shape[0]

				if self.kmeans_work is None:
					self.kmeans_work_init(m, k, n)

				# TODO: check type of settings
				if settings is None:
					settings = ClusteringSettings(m)

				lib.kmeans_easy_run(self.kmeans_work.pointer, settings.pointer,
									self.io)

				u = UpsamplingVector(indices=self.a2c)
				u.clean_assignments()
				k_out = u.max_assignment + 1

				C *= 0
				for idx_a, idx_c in enumerate(u.indices):
					C[idx_c, :] += A[idx_a, :]

				for i, ct in enumerate(u.counts[:k_out]):
					C[i, :] *= 1. / ct

				return C[:k_out, :], u.indices, u.counts[:k_out]

			def kmeans(self, A, k, assignments, settings=None):
				C = zeros((k, A.shape[1]), dtype=lib.pyfloat)
				return self.kmeans_inplace(A, C, assignments,
										   settings=settings)

			def blockwise_kmeans_inplace(self, A_blocks, C_blocks,
										 assignment_blocks, settings_blocks=None):

				max_A = array([A.shape[0] for A in A_blocks]).max()
				max_C = array([C.shape[0] for C in C_blocks]).max()
				n = A.shape[1]

				assignments_final = []
				counts_final = []
				C_final = []

				self.kmeans_work_init(max_A, max_C, n)

				if settings_blocks is None:
					settings_blocks = [ClusteringSettings(A.shape[0]) for A in A_blocks]

				for b in xrange(len(A_blocks)):
					A = A_blocks[b]
					C = C_blocks[b]
					assignments = assignment_blocks[b]

					self.kmeans_work.resize(A.shape[0], C.shape[0])

					C_out, a2c_out, counts_out = self.kmeans_inplace(
							A, C, assignments, settings=settings_blocks[b])

					assignments_final.append(a2c_out)
					counts_final.append(counts_out)
					C_final.append(C_out)

				return (C_final, assignments_final, counts_final)

			def blockwise_kmeans(self, A_blocks, assignment_blocks,
								 settings_blocks=None):

				C_blocks = []
				for b in xrange(len(A_blocks)):
					size1 = assignment_blocks[b].max() + 1
					size2 = A_blocks[b].shape[1]
					C_blocks.append(zeros((size1, size2), dtype=lib.pyfloat))

				return self.blockwise_kmeans_inplace(
						A_blocks, C_blocks, assignment_blocks,
						settings_blocks=settings_blocks)

		self.Clustering = Clustering
