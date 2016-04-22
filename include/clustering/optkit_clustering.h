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
static ok_status assign_clusters_l2(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h);
static ok_status assign_clusters_l2_lInf_cap(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h, ok_float maxdist);

/* COMMON IMPLEMENTATION */
static ok_status cluster_aid_subselect(cluster_aid * h, size_t offset_A,
	size_t offset_C, size_t sub_size_A, size_t sub_size_C)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!h || !h->indicator)
		return OPTKIT_ERROR_UNALLOCATED;

	h->reassigned = 0;

	OPTKIT_RETURN_IF_ERROR(upsamplingvec_subvector(&h->a2c_tentative,
		&h->a2c_tentative_full, offset_A, sub_size_A, sub_size_C));

	vector_subvector(&h->c_squared, &h->c_squared_full, offset_C,
		sub_size_C);
	vector_subvector(&h->d_min, &h->d_min_full, offset_A, sub_size_A);
	matrix_submatrix(&h->D, &h->D_full, offset_A, offset_C, sub_size_C,
		sub_size_A);
	return err;
}

ok_status cluster_aid_alloc(cluster_aid * h, size_t size_A, size_t size_C,
	enum CBLAS_ORDER order)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!h)
		return OPTKIT_ERROR_UNALLOCATED;
	else if (h->indicator)
		return OPTKIT_ERROR_OVERWRITE;

	/* clear h */
	memset(h, 0, sizeof(*h));

	h->indicator = (int *) malloc(sizeof(int));
	OPTKIT_RETURN_IF_ERROR(upsamplingvec_alloc(&h->a2c_tentative_full,
		size_A, size_C));
	vector_calloc(&h->c_squared_full, size_C);
	vector_calloc(&h->d_min_full, size_A);
	matrix_calloc(&h->D_full, size_C, size_A, order);
	OPTKIT_CHECK_ERROR(&err, blas_make_handle(&h->hdl));
	OPTKIT_CHECK_ERROR(&err, cluster_aid_subselect(h, 0, 0, size_A, size_C));
	return err;
}

ok_status cluster_aid_free(cluster_aid * h)
{
	ok_status err;
	if (!h || !h->indicator)
		return OPTKIT_ERROR_UNALLOCATED;

	blas_destroy_handle(h->hdl);
	if (h->A_reducible.data)
		matrix_free(&h->A_reducible);
	matrix_free(&h->D_full);
	vector_free(&h->d_min_full);
	vector_free(&h->c_squared_full);
	OPTKIT_CHECK_ERROR(&err, upsamplingvec_free(&h->a2c_tentative_full));
	ok_free(h->indicator);

	/* clear h */
	memset(h, 0, sizeof(*h));
	return err;
}

ok_status kmeans_work_alloc(kmeans_work * w, size_t n_vectors,
	size_t n_clusters, size_t vec_length)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!w)
		return OPTKIT_ERROR_UNALLOCATED;
	else if (w->indicator)
		return OPTKIT_ERROR_OVERWRITE;

	memset(w, 0, sizeof(*w));
	w->indicator = (int *) malloc(sizeof(int));
	w->n_vectors = n_vectors;
	w->n_clusters = n_clusters;
	w->vec_length = vec_length;
	matrix_alloc(&w->A, n_vectors, vec_length, CblasRowMajor);
	matrix_alloc(&w->C, n_clusters, vec_length, CblasRowMajor);
	upsamplingvec_alloc(&w->a2c, n_vectors, n_clusters);
	vector_alloc(&w->counts, n_clusters);
	cluster_aid_alloc(&w->h, n_vectors, n_clusters, CblasRowMajor);
	return err;
}

ok_status kmeans_work_free(kmeans_work * w)
{
	if (!w || !w->indicator)
		return OPTKIT_ERROR_UNALLOCATED;

	matrix_free(&w->A);
	matrix_free(&w->C);
	upsamplingvec_free(&w->a2c);
	vector_free(&w->counts);
	cluster_aid_free(&w->h);
	ok_free(w->indicator);
	memset(w, 0, sizeof(*w));
	return OPTKIT_SUCCESS;
}

ok_status kmeans_work_subselect(kmeans_work * w, size_t n_subvectors,
	size_t n_subclusters, size_t subvec_length)
{
	if (!w || !w->indicator)
		return OPTKIT_ERROR_UNALLOCATED;

	if (n_subvectors > w->n_vectors || n_subclusters > w->n_clusters ||
		subvec_length > w->vec_length)
		return OPTKIT_ERROR_OUT_OF_BOUNDS;

	w->A.size1 = n_subvectors;
	w->A.size2 = subvec_length;
	w->C.size1 = n_subclusters;
	w->C.size2 = subvec_length;
	w->a2c.size1 = n_subvectors;
	w->a2c.size2 = n_subclusters;
	w->a2c.vec.size = n_subvectors;
	w->counts.size = n_subclusters;
	return cluster_aid_subselect(&w->h, 0, 0, n_subvectors, n_subclusters);
}

ok_status kmeans_work_load(kmeans_work * w, ok_float * A,
	const enum CBLAS_ORDER orderA, ok_float * C,
	const enum CBLAS_ORDER orderC, size_t * a2c, size_t stride_a2c,
	ok_float * counts, size_t stride_counts)
{
	if (!w || !w->indicator || !A || !C || !a2c || !counts)
		return OPTKIT_ERROR_UNALLOCATED;

	matrix_memcpy_ma(&w->A, A, orderA);
	matrix_memcpy_ma(&w->C, C, orderC);
	indvector_memcpy_va(&w->a2c.vec, a2c, stride_a2c);
	vector_memcpy_va(&w->counts, counts, stride_counts);
	return OPTKIT_SUCCESS;
}

ok_status kmeans_work_extract(ok_float * C,
	const enum CBLAS_ORDER orderC, size_t * a2c, size_t stride_a2c,
	ok_float * counts, size_t stride_counts, kmeans_work * w)
{
	if (!w || !w->indicator || !C || !a2c || !counts)
		return OPTKIT_ERROR_UNALLOCATED;

	matrix_memcpy_am(C, &w->C, orderC);
	indvector_memcpy_av(a2c, &w->a2c.vec, stride_a2c);
	vector_memcpy_av(counts, &w->counts, stride_counts);
	return OPTKIT_SUCCESS;
}

ok_status cluster(matrix * A, matrix * C, upsamplingvec * a2c,
	cluster_aid ** helper, ok_float maxdist)
{
	ok_status err = OPTKIT_SUCCESS;
	int cluster_aid_provided = (helper != OK_NULL);
	cluster_aid * h;

	if (!cluster_aid_provided) {
		h = *helper;
	} else {
		h = (cluster_aid *) malloc(sizeof(*h));
		OPTKIT_RETURN_IF_ERROR(
			cluster_aid_alloc(h, A->size1, C->size1, A->order));
		*helper = h;
	}

	/*
	 * Prep work: set
	 *	h->c_squared_k = c_k'c_k,
	 *	D_ki = - 2 * c_k'a_i
	 */
	linalg_matrix_row_squares(CblasNoTrans, C, &h->c_squared);
	blas_gemm(h->hdl, CblasNoTrans, CblasTrans, -2 * kOne, C, A, kZero,
		&h->D);

	/* Form D_{ki} = - 2 * c_k'a_i + c_k^2 */
	linalg_matrix_broadcast_vector(&h->D, &h->c_squared, OkTransformAdd,
		CblasLeft);

	/* set tentative cluster assigment of vector i argmin_k {D_ki} */
	linalg_matrix_reduce_indmin(&h->a2c_tentative.vec, &h->d_min, &h->D,
		CblasLeft);

	/* finalize cluster assignements */
	if (maxdist == OK_INFINITY)
		OPTKIT_CHECK_ERROR(&err, assign_clusters_l2(A, C, a2c, h));
	else
		OPTKIT_CHECK_ERROR(&err, assign_clusters_l2_lInf_cap(A, C, a2c,
			h, maxdist));
	return err;
}

/*
 * given matrices A and C, and upsampling linear operator U, set
 *
 *	C = diag(U'1)^{-1} * U'A
 */
ok_status calculate_centroids(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts)
{
	ok_status err = OPTKIT_SUCCESS;
	OPTKIT_CHECK_ERROR(&err, upsamplingvec_mul_matrix(CblasTrans,
		CblasNoTrans, CblasNoTrans, kOne, a2c, A, kZero, C));
	OPTKIT_CHECK_ERROR(&err, upsamplingvec_count(a2c, counts));
	vector_safe_recip(counts);
	linalg_matrix_broadcast_vector(C, counts, OkTransformScale, CblasLeft);
	return err;
}

/*
 * interpolate between maxA and reltol * maxA: letting
 *
 *	alpha = iter / max_iter,
 *
 * return
 *
 *	(alpha * reltol + (1 - alpha)) maxA = (1 + alpha (reltol - 1)) maxA
 */
static void get_distance_tolerance(ok_float *tol, const ok_float * maxA,
	const ok_float * reltol, const size_t * iter, const size_t * maxiter)
{
	if ((*iter) == 0)
		*tol = OK_INFINITY;
	else
		*tol = (kOne + ((ok_float) (*iter) / (*maxiter)) *
			(*reltol - kOne)) * (*maxA);
}

static ok_status k_means_finish(cluster_aid * h)
{
	ok_status err = cluster_aid_free(h);
	if (h)
		ok_free(h);
	return err;
}

ok_status k_means(matrix * A, matrix * C, upsamplingvec * a2c, vector * counts,
	cluster_aid * h, const ok_float dist_reltol, const size_t change_abstol,
	const size_t maxiter, const uint verbose)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t iter;
	vector vA;
	ok_float tol, rowmax, maxA = -OK_FLOAT_MAX;
	char iterfmt[] =
		"iteration: %u \t changed: %u \t change tol: %u\n";
	char exitfmt[] = "quitting k-means after %u iterations, error = %u\n";
	const int cluster_aid_provided = (h != OK_NULL);
	int valid;

	/* bounds/dimension checks */
	OPTKIT_RETURN_IF_ERROR(upsamplingvec_check_bounds(a2c));
	valid = (counts->size == C->size1) && (A->size2 == C->size2);
	valid &= (A->size1 == a2c->size1) && (C->size2 >= a2c->size2);
	if (!valid)
		return OPTKIT_ERROR_DIMENSION_MISMATCH;

	/* calculate max entry of A */
	for (iter = 0; iter < A->size1 && !err; ++iter) {
		matrix_row(&vA, A, iter);
		rowmax = vector_max(&vA);
		maxA = maxA > rowmax ? maxA : rowmax;
	}

	/* ensure C is initialized */
	OPTKIT_CHECK_ERROR(&err, calculate_centroids(A, C, a2c, counts));

	if (!err && verbose)
		printf("\nstarting k-means on %zu vectors and %zu centroids\n",
			A->size1, C->size1);

	for (iter = 0; iter < maxiter && !err; ++iter) {
		get_distance_tolerance(&tol, &maxA, &dist_reltol, &iter,
			&maxiter);
		printf("%s %zu %s %f\n", "ITER", iter, "DISTANCE TOLERANCE",
			tol);
		OPTKIT_CHECK_ERROR(&err, cluster(A, C, a2c, &h, tol));
		OPTKIT_CHECK_ERROR(&err, calculate_centroids(A, C, a2c,
			counts));
		if (verbose)
			printf(iterfmt, iter, h->reassigned, change_abstol);
		if (h->reassigned < change_abstol)
			break;
	}
	if (!err && verbose)
		printf(exitfmt, iter + 1, err);

	if (!cluster_aid_provided)
		OPTKIT_MAX_ERROR(&err, k_means_finish(h));

	return err;
}

void * kmeans_easy_init(size_t n_vectors, size_t n_clusters, size_t vec_length)
{
	kmeans_work * w;
	ok_alloc(w, kmeans_work, sizeof(*w));
	kmeans_work_alloc(w, n_vectors, n_clusters, vec_length);
	return (void * ) w;
}

ok_status kmeans_easy_resize(const void * work, size_t n_vectors,
	size_t n_clusters, size_t vec_length)
{
	return kmeans_work_subselect((kmeans_work *) work, n_vectors,
		n_clusters, vec_length);
}

ok_status kmeans_easy_run(const void * work, const kmeans_settings * const s,
	const kmeans_io * io)
{
	ok_status err;
	if (!work || !s)
		return OPTKIT_ERROR_UNALLOCATED;
	kmeans_work * w = (kmeans_work *) work;
	if (!w || !w->indicator)
		return OPTKIT_ERROR_UNALLOCATED;

	OPTKIT_CHECK_ERROR(&err, kmeans_work_load(w, io->A, io->orderA, io->C,
		io->orderC, io->a2c, io->stride_a2c, io->counts,
		io->stride_counts));

	OPTKIT_CHECK_ERROR(&err, k_means(&w->A, &w->C, &w->a2c, &w->counts,
		&w->h, s->dist_reltol, s->change_abstol, s->maxiter,
		s->verbose));

	OPTKIT_CHECK_ERROR(&err, kmeans_work_extract(io->C, io->orderC,
		io->a2c, io->stride_a2c, io->counts, io->stride_counts, w));

	return err;
}
ok_status kmeans_easy_finish(void * work)
{
	ok_status err = kmeans_work_free((kmeans_work *) work);
	ok_free(work);
	return err;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CLUSTERING_H_ */
