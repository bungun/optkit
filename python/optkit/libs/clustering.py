from ctypes import c_int, c_uint, c_size_t, c_void_p, CFUNCTYPE, POINTER, \
				   Structure
from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_sparse_linsys_ctypes, attach_base_ccalls, attach_vector_ccalls, \
	attach_dense_linsys_ccalls, attach_sparse_linsys_ccalls

class ClusteringLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libcluster_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_dense_linsys_ccalls)
		self.attach_calls.append(attach_clustering_ctypes)
		self.attach_calls.append(attach_clustering_ccalls)

def attach_clustering_ctypes(lib, single_precision=False):
	if not 'matrix' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	c_size_t_p = lib.c_size_t_p
	vector = lib.vector
	indvector = lib.indvector
	matrix = lib.matrix
	matrix_p = lib.matrix_p

	class upsamplingvec(Structure):
		_fields_ = [('size1', c_size_t),
					('size2', c_size_t),
					('stride', c_size_t),
					('indices', c_size_t_p),
					('vec', indvector)]

	lib.upsamplingvec = upsamplingvec
	lib.upsamplingvec_p = POINTER(lib.upsamplingvec)

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

	lib.cluster_aid = cluster_aid
	lib.cluster_aid_p = POINTER(lib.cluster_aid)

	class kmeans_work(Structure):
		_fields_ = [('indicator', POINTER(c_int)),
					('n_vectors', c_size_t),
					('n_clusters', c_size_t),
					('vec_length', c_size_t),
					('A', matrix),
					('C', matrix),
					('a2c', upsamplingvec),
					('counts', vector),
					('h', cluster_aid)]

	lib.kmeans_work = kmeans_work
	lib.kmeans_work_p = POINTER(lib.kmeans_work)

	class kmeans_settings(Structure):
		_fields_ = [('dist_reltol', ok_float),
					('change_abstol', c_size_t),
					('maxiter', c_size_t),
					('verbose', c_uint)]

	lib.kmeans_settings = kmeans_settings
	lib.kmeans_settings_p = POINTER(lib.kmeans_settings)

	class kmeans_io(Structure):
		_fields_ = [('A', ok_float_p),
					('C', ok_float_p),
					('counts', ok_float_p),
					('a2c', c_size_t_p),
					('orderA', c_uint),
					('orderC', c_uint),
					('stride_a2c', c_uint),
					('stride_counts', c_size_t)]

	lib.kmeans_io = kmeans_io
	lib.kmeans_io_p = POINTER(lib.kmeans_io)

def attach_clustering_ccalls(lib, single_precision=False):
	if not 'matrix_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'kmeans_work_p' in lib.__dict__:
		attach_cluster_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	c_size_t_p = lib.c_size_t_p
	indvector = lib.indvector
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p
	upsamplingvec_p = lib.upsamplingvec_p
	cluster_aid_p = lib.cluster_aid_p
	kmeans_work_p = lib.kmeans_work_p
	kmeans_settings_p = lib.kmeans_settings_p
	kmeans_io_p = lib.kmeans_io_p

	# argument types
	lib.upsamplingvec_alloc.argtypes = [upsamplingvec_p, c_size_t,
										c_size_t]
	lib.upsamplingvec_free.argtypes = [upsamplingvec_p]
	lib.upsamplingvec_check_bounds.argtypes = [upsamplingvec_p]
	lib.upsamplingvec_update_size.argtypes = [upsamplingvec_p]
	lib.upsamplingvec_subvector.argtypes = [upsamplingvec_p,
											upsamplingvec_p, c_size_t,
											c_size_t, c_size_t]
	lib.upsamplingvec_mul_matrix.argtypes = [c_uint, c_uint, c_uint,
											 ok_float, upsamplingvec_p,
											 matrix_p, ok_float,
											 matrix_p]
	lib.upsamplingvec_count.argtypes = [upsamplingvec_p, vector_p]
	lib.cluster_aid_alloc.argtypes = [cluster_aid_p, c_size_t,
									  c_size_t, c_uint]
	lib.cluster_aid_free.argtypes = [cluster_aid_p]
	lib.kmeans_work_alloc.argtypes = [kmeans_work_p, c_size_t,
									  c_size_t, c_size_t]
	lib.kmeans_work_free.argtypes = [kmeans_work_p]
	lib.kmeans_work_subselect.argtypes = [kmeans_work_p, c_size_t,
										  c_size_t, c_size_t]
	lib.kmeans_work_load.argtypes = [kmeans_work_p, ok_float_p,
									 c_uint, ok_float_p, c_uint,
									 c_size_t_p, c_size_t, ok_float_p,
									 c_size_t]
	lib.kmeans_work_extract.argtypes = [ok_float_p, c_uint,
										c_size_t_p, c_size_t,
										ok_float_p, c_size_t,
										kmeans_work_p]
	lib.cluster.argtypes = [matrix_p, matrix_p, upsamplingvec_p,
							POINTER(cluster_aid_p), ok_float]
	lib.calculate_centroids.argtypes = [matrix_p, matrix_p,
										upsamplingvec_p, vector_p]
	lib.k_means.argtypes = [matrix_p, matrix_p, upsamplingvec_p,
							vector_p, cluster_aid_p, ok_float,
							c_size_t, c_size_t, c_uint]
	lib.kmeans_easy_init.argtypes = [c_size_t, c_size_t, c_size_t]
	lib.kmeans_easy_resize.argtypes = [c_void_p, c_size_t, c_size_t,
										c_size_t]
	lib.kmeans_easy_run.argtypes = [c_void_p, kmeans_settings_p,
									 kmeans_io_p]
	lib.kmeans_easy_finish.argtypes = [c_void_p]

	# return types
	lib.upsamplingvec_alloc.restype = c_uint
	lib.upsamplingvec_free.restype = c_uint
	lib.upsamplingvec_check_bounds.restype = c_uint
	lib.upsamplingvec_update_size.restype = c_uint
	lib.upsamplingvec_subvector.restype = c_uint
	lib.upsamplingvec_mul_matrix.restype = c_uint
	lib.upsamplingvec_count.restype = c_uint
	lib.cluster_aid_alloc.restype = c_uint
	lib.cluster_aid_free.restype = c_uint
	lib.kmeans_work_alloc.restype = c_uint
	lib.kmeans_work_free.restype = c_uint
	lib.kmeans_work_subselect.restype = c_uint
	lib.kmeans_work_load.restype = c_uint
	lib.kmeans_work_extract.restype = c_uint
	lib.cluster.restype = c_uint
	lib.calculate_centroids.restype = c_uint
	lib.k_means.restype = c_uint
	lib.kmeans_easy_init.restype = c_void_p
	lib.kmeans_easy_resize.restype = c_uint
	lib.kmeans_easy_run.restype = c_uint
	lib.kmeans_easy_finish.restype = c_uint
