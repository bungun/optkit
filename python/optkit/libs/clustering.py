from ctypes import c_int, c_uint, c_size_t, c_void_p, CFUNCTYPE, POINTER, \
				   Structure
from optkit.libs.loader import retrieve_libs, validate_lib

class ClusteringLibs(object):
	def __init__(self):
		self.libs, search_results = retrieve_libs('libcluster_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, denselib, single_precision=False, gpu=False):
		device = 'gpu' if gpu else 'cpu'
		precision = '32' if single_precision else '64'
		lib_key = '{}{}'.format(device, precision)

		if lib_key not in self.libs:
			return None
		elif self.libs[lib_key] is None:
			return None

		validate_lib(denselib, 'denselib', 'vector_calloc', type(self),
			single_precision, gpu)

		lib = self.libs[lib_key]
		if lib.INITIALIZED:
			return lib
		else:
			ok_float = denselib.ok_float
			c_size_t_p = denselib.c_size_t_p
			vector = denselib.vector
			vector_p = denselib.vector_p
			indvector = denselib.indvector
			indvector_p = denselib.indvector_p
			matrix = denselib.matrix
			matrix_p = denselib.matrix_p

			class upsamplingvec(Structure):
				_fields_ = [('size1', c_size_t),
							('size2', c_size_t),
							('stride', c_size_t),
							('indices', c_size_t_p),
							('vec', indvector)]

			upsamplingvec_p = POINTER(upsamplingvec)
			lib.upsamplingvec = upsamplingvec
			lib.upsamplingvec_p = upsamplingvec_p

			class cluster_aid(Structure):
				_fields_ = [('indicator', POINTER(c_int)),
							('hdl', c_void_p),
							('a2c_tentative_full', upsamplingvec),
							('a2c_tentative', upsamplingvec),
							('d_min_full', vector),
							('d_min', vector),
							('c_squared_full', vector),
							('c_squared', vector),
							('D_full', matrix),
							('D', matrix),
							('A_reducible', matrix),
							('reassigned', c_size_t)]

			cluster_aid_p = POINTER(cluster_aid)
			lib.cluster_aid = cluster_aid
			lib.cluster_aid_p = cluster_aid_p

			class cluster_work(Structure):
				_fields_ = [('indicator', POINTER(c_int)),
							('n_vectors', c_size_t),
							('n_clusters', c_size_t),
							('vec_length', c_size_t),
							('A', matrix),
							('C', matrix),
							('a2c', upsamplingvec),
							('counts', vector),
							('h', cluster_aid)]

			cluster_work_p = POINTER(cluster_work)
			lib.cluster_work = cluster_work
			lib.cluster_work_p = cluster_work_p


			# argument types
			lib.upsamplingvec_alloc.argtypes = [upsamplingvec_p, c_size_t,
												c_size_t]
			lib.upsamplingvec_free.argtypes = [upsamplingvec_p]
			lib.upsamplingvec_check_bounds.argtypes = [upsamplingvec_p]
			lib.upsamplingvec_subvector.argtypes = [upsamplingvec_p,
													upsamplingvec_p, c_size_t,
													c_size_t, c_size_t,
													c_size_t]
			lib.upsamplingvec_mul_matrix.argtypes = [c_uint, c_uint, c_uint,
													 ok_float, upsamplingvec_p,
													 matrix_p, ok_float,
													 matrix_p]
			lib.upsamplingvec_count.argtypes = [upsamplingvec_p, vector_p]
			lib.upsamplingvec_shift.argtypes = [upsamplingvec_p, c_size_t,
												c_uint]
			lib.cluster_aid_alloc.argtypes = [cluster_aid_p, c_size_t,
											  c_size_t, c_uint]
			lib.cluster_aid_free.argtypes = [cluster_aid_p]
			lib.cluster.argtypes = [matrix_p, matrix_p, upsamplingvec_p,
									POINTER(cluster_aid_p), ok_float]
			lib.calculate_centroids.argtypes = [matrix_p, matrix_p,
												upsamplingvec_p, vector_p]
			lib.k_means.argtypes = [matrix_p, matrix_p, upsamplingvec_p,
									vector_p, cluster_aid_p, ok_float,
									c_size_t, c_size_t, c_uint]

			# return types
			lib.upsamplingvec_alloc.restype = c_uint
			lib.upsamplingvec_free.restype = c_uint
			lib.upsamplingvec_check_bounds.restype = c_uint
			lib.upsamplingvec_subvector.restype = c_uint
			lib.upsamplingvec_mul_matrix.restype = c_uint
			lib.upsamplingvec_count.restype = c_uint
			lib.upsamplingvec_shift.restype = c_uint
			lib.cluster_aid_alloc.restype = c_uint
			lib.cluster_aid_free.restype = c_uint
			lib.cluster.restype = c_uint
			lib.calculate_centroids.restype = c_uint
			lib.k_means.restype = c_uint

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib