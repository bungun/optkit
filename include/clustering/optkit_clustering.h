#ifndef OPTKIT_CLUSTERING_CLUSTERING_H_
#define OPTKIT_CLUSTERING_CLUSTERING_H_

#include "optkit_dense.h"
#include "optkit_upsampling_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cluster_aid {
	int * indicator;
	void * hdl; /* linalg handle */
	upsamplingvec a2c_tentative_full, a2c_tentative;
	vector d_min_full, d_min, c_squared_full, c_squared;
	matrix D_full, D, A_reducible;
	size_t reassigned;
} cluster_aid;

typedef struct kmeans_work {
	int * indicator;
	size_t n_vectors, n_clusters, vec_length;
	matrix A, C;
	upsamplingvec a2c;
	vector counts;
	cluster_aid h;
} kmeans_work;

typedef struct kmeans_settings {
	ok_float dist_reltol;
	size_t change_abstol, maxiter;
	uint verbose;
} kmeans_settings;

typedef struct kmeans_io {
	ok_float * A, * C, * counts;
	size_t * a2c;
	enum CBLAS_ORDER orderA, orderC;
	size_t stride_a2c, stride_counts;
} kmeans_io;

/* CPU/GPU-SPECIFIC IMPLEMENTATION */
ok_status assign_clusters_l2(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h);
ok_status assign_clusters_l2_lInf_cap(matrix * A,
	matrix * C, upsamplingvec * a2c, cluster_aid * h, ok_float maxdist);

/* COMMON IMPLEMENTATION */

ok_status cluster_aid_alloc(cluster_aid * h, size_t size_A, size_t size_C,
	enum CBLAS_ORDER order);
ok_status cluster_aid_free(cluster_aid * h);
static ok_status cluster_aid_subselect(cluster_aid * h, size_t offset_A,
	size_t offset_C, size_t sub_size_A, size_t sub_size_C);

ok_status kmeans_work_alloc(kmeans_work * w, size_t n_vectors,
	size_t n_clusters, size_t vec_length);
ok_status kmeans_work_free(kmeans_work * w);
ok_status kmeans_work_subselect(kmeans_work * w, size_t n_subvectors,
	size_t n_subclusters, size_t subvec_length);

ok_status kmeans_work_load(kmeans_work * w, ok_float * A,
	const enum CBLAS_ORDER orderA, ok_float * C,
	const enum CBLAS_ORDER orderC, size_t * a2c, size_t stride_a2c,
	ok_float * counts, size_t stride_counts);
ok_status kmeans_work_extract(ok_float * C,
	const enum CBLAS_ORDER orderC, size_t * a2c, size_t stride_a2c,
	ok_float * counts, size_t stride_counts, kmeans_work * w);
ok_status cluster(matrix * A, matrix * C, upsamplingvec * a2c,
	cluster_aid * helper, ok_float maxdist);
ok_status calculate_centroids(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts);

static void get_distance_tolerance(ok_float *tol, const ok_float * maxA,
	const ok_float * reltol, const size_t * iter, const size_t * maxiter);
static ok_status k_means_finish(cluster_aid * h);

ok_status k_means(matrix * A, matrix * C, upsamplingvec * a2c, vector * counts,
	cluster_aid * h, const ok_float dist_reltol, const size_t change_abstol,
	const size_t maxiter, const uint verbose);
void * kmeans_easy_init(size_t n_vectors, size_t n_clusters, size_t vec_length);
ok_status kmeans_easy_resize(const void * work, size_t n_vectors,
	size_t n_clusters, size_t vec_length);
ok_status kmeans_easy_run(const void * work, const kmeans_settings * const s,
	const kmeans_io * io);
ok_status kmeans_easy_finish(void * work);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CLUSTERING_H_ */
