from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.linsys import include_ok_dense, ok_dense_API


def include_ok_clustering(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'kmeans_io_p', attach_clustering_ctypes, **include_args)

def ok_cluster_API(): return ok_dense_API() + [attach_clustering_ccalls]

class ClusteringLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libcluster_', ok_cluster_API())

def attach_clustering_ctypes(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	c_size_t_p = lib.c_size_t_p
	vector = lib.vector
	indvector = lib.indvector
	matrix = lib.matrix
	matrix_p = lib.matrix_p

	class upsamplingvec(ct.Structure):
		_fields_ = [('size1', ct.c_size_t),
					('size2', ct.c_size_t),
					('stride', ct.c_size_t),
					('indices', c_size_t_p),
					('vec', indvector)]

	lib.upsamplingvec = upsamplingvec
	lib.upsamplingvec_p = ct.POINTER(lib.upsamplingvec)

	class cluster_aid(ct.Structure):
		_fields_ = [('hdl', ct.c_void_p),
					('a2c_tentative_full', upsamplingvec),
					('a2c_tentative', upsamplingvec),
					('d_min_full', vector),
					('d_min', vector),
					('c_squared_full', vector),
					('c_squared', vector),
					('D_full', matrix),
					('D', matrix),
					('A_reducible', matrix),
					('reassigned', ct.c_size_t)]

	lib.cluster_aid = cluster_aid
	lib.cluster_aid_p = ct.POINTER(lib.cluster_aid)

	class kmeans_work(ct.Structure):
		_fields_ = [('n_vectors', ct.c_size_t),
					('n_clusters', ct.c_size_t),
					('vec_length', ct.c_size_t),
					('A', matrix),
					('C', matrix),
					('a2c', upsamplingvec),
					('counts', vector),
					('h', cluster_aid)]

	lib.kmeans_work = kmeans_work
	lib.kmeans_work_p = ct.POINTER(lib.kmeans_work)

	class kmeans_settings(ct.Structure):
		_fields_ = [('dist_reltol', ok_float),
					('change_abstol', ct.c_size_t),
					('maxiter', ct.c_size_t),
					('verbose', ct.c_uint)]

	lib.kmeans_settings = kmeans_settings
	lib.kmeans_settings_p = ct.POINTER(lib.kmeans_settings)

	class kmeans_io(ct.Structure):
		_fields_ = [('A', ok_float_p),
					('C', ok_float_p),
					('counts', ok_float_p),
					('a2c', c_size_t_p),
					('orderA', ct.c_uint),
					('orderC', ct.c_uint),
					('stride_a2c', ct.c_uint),
					('stride_counts', ct.c_size_t)]

	lib.kmeans_io = kmeans_io
	lib.kmeans_io_p = ct.POINTER(lib.kmeans_io)

def attach_clustering_ccalls(lib, single_precision=False):
	include_ok_clustering(lib, single_precision=single_precision)

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
	lib.upsamplingvec_alloc.argtypes = [
			upsamplingvec_p, ct.c_size_t, ct.c_size_t]
	lib.upsamplingvec_free.argtypes = [upsamplingvec_p]
	lib.upsamplingvec_check_bounds.argtypes = [upsamplingvec_p]
	lib.upsamplingvec_update_size.argtypes = [upsamplingvec_p]
	lib.upsamplingvec_subvector.argtypes = [
			upsamplingvec_p, upsamplingvec_p, ct.c_size_t, ct.c_size_t,
			ct.c_size_t]
	lib.upsamplingvec_mul_matrix.argtypes = [
			ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint, ok_float,
			upsamplingvec_p, matrix_p, ok_float, matrix_p]
	lib.upsamplingvec_count.argtypes = [upsamplingvec_p, vector_p]
	lib.cluster_aid_alloc.argtypes = [
			cluster_aid_p, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.cluster_aid_free.argtypes = [cluster_aid_p]
	lib.kmeans_work_alloc.argtypes = [
			kmeans_work_p, ct.c_size_t, ct.c_size_t, ct.c_size_t]
	lib.kmeans_work_free.argtypes = [kmeans_work_p]
	lib.kmeans_work_subselect.argtypes = [
			kmeans_work_p, ct.c_size_t, ct.c_size_t, ct.c_size_t]
	lib.kmeans_work_load.argtypes = [
			kmeans_work_p, ok_float_p, ct.c_uint, ok_float_p, ct.c_uint,
			c_size_t_p, ct.c_size_t, ok_float_p, ct.c_size_t]
	lib.kmeans_work_extract.argtypes = [
			ok_float_p, ct.c_uint, c_size_t_p, ct.c_size_t, ok_float_p,
			ct.c_size_t, kmeans_work_p]
	lib.cluster.argtypes = [
			matrix_p, matrix_p, upsamplingvec_p, cluster_aid_p, ok_float]
	lib.calculate_centroids.argtypes = [
			matrix_p, matrix_p, upsamplingvec_p, vector_p, cluster_aid_p]
	lib.k_means.argtypes = [
			matrix_p, matrix_p, upsamplingvec_p, vector_p, cluster_aid_p,
			ok_float, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.kmeans_easy_init.argtypes = [ct.c_size_t, ct.c_size_t, ct.c_size_t]
	lib.kmeans_easy_resize.argtypes = [
			ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_size_t]
	lib.kmeans_easy_run.argtypes = [
			ct.c_void_p, kmeans_settings_p, kmeans_io_p]
	lib.kmeans_easy_finish.argtypes = [ct.c_void_p]

	# return types
	OptkitLibs.attach_default_restype(
			lib.upsamplingvec_alloc,
			lib.upsamplingvec_free,
			lib.upsamplingvec_check_bounds,
			lib.upsamplingvec_update_size,
			lib.upsamplingvec_subvector,
			lib.upsamplingvec_mul_matrix,
			lib.upsamplingvec_count,
			lib.cluster_aid_alloc,
			lib.cluster_aid_free,
			lib.kmeans_work_alloc,
			lib.kmeans_work_free,
			lib.kmeans_work_subselect,
			lib.kmeans_work_load,
			lib.kmeans_work_extract,
			lib.cluster,
			lib.calculate_centroids,
			lib.k_means,
			lib.kmeans_easy_resize,
			lib.kmeans_easy_run,
			lib.kmeans_easy_finish,
	)
	lib.kmeans_easy_init.restype = ct.c_void_p

