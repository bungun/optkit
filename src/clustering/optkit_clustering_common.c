#include "optkit_clustering.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status cluster_aid_alloc(cluster_aid * h, size_t size_A, size_t size_C,
	enum CBLAS_ORDER order)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(h);

	/* clear h */
	memset(h, 0, sizeof(*h));

	OK_CHECK_ERR( err,
		upsamplingvec_alloc(&h->a2c_tentative_full, size_A, size_C) );
	OK_CHECK_ERR( err,
		vector_calloc(&h->c_squared_full, size_C) );
	OK_CHECK_ERR( err,
		vector_calloc(&h->d_min_full, size_A) );
	OK_CHECK_ERR( err,
		matrix_calloc(&h->D_full, size_C, size_A, order) );
	OK_CHECK_ERR( err,
		blas_make_handle(&h->hdl) );
	OK_CHECK_ERR( err,
		cluster_aid_subselect(h, 0, 0, size_A, size_C) );
	if (err)
		OK_MAX_ERR( err, cluster_aid_free(h) );
	return err;
}

ok_status cluster_aid_free(cluster_aid * h)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(h);

	OK_MAX_ERR( err, blas_destroy_handle(h->hdl) );
	if (h->A_reducible.data)
		OK_MAX_ERR( err, matrix_free(&h->A_reducible) );
	OK_MAX_ERR( err, matrix_free(&h->D_full) );
	OK_MAX_ERR( err, vector_free(&h->d_min_full) );
	OK_MAX_ERR( err, vector_free(&h->c_squared_full) );
	OK_MAX_ERR( err, upsamplingvec_free(&h->a2c_tentative_full) );

	/* clear h */
	memset(h, 0, sizeof(*h));
	return err;
}

static ok_status cluster_aid_subselect(cluster_aid * h, size_t offset_A,
	size_t offset_C, size_t sub_size_A, size_t sub_size_C)
{
	OK_CHECK_PTR(h);

	h->reassigned = 0;

	OK_RETURNIF_ERR( upsamplingvec_subvector(&h->a2c_tentative,
		&h->a2c_tentative_full, offset_A, sub_size_A, sub_size_C) );

	OK_RETURNIF_ERR( vector_subvector(&h->c_squared, &h->c_squared_full,
		offset_C, sub_size_C) );
	OK_RETURNIF_ERR( vector_subvector(&h->d_min, &h->d_min_full, offset_A,
		sub_size_A) );
	return matrix_submatrix(&h->D, &h->D_full, offset_A, offset_C,
		sub_size_C, sub_size_A);
}

ok_status kmeans_work_alloc(kmeans_work * w, size_t n_vectors,
	size_t n_clusters, size_t vec_length)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(w);

	memset(w, 0, sizeof(*w));
	w->n_vectors = n_vectors;
	w->n_clusters = n_clusters;
	w->vec_length = vec_length;
	OK_CHECK_ERR( err,
		matrix_calloc(&w->A, n_vectors, vec_length, CblasRowMajor) );
	OK_CHECK_ERR( err,
		matrix_calloc(&w->C, n_clusters, vec_length, CblasRowMajor) );
	OK_CHECK_ERR( err,
		upsamplingvec_alloc(&w->a2c, n_vectors, n_clusters) );
	OK_CHECK_ERR( err,
		vector_calloc(&w->counts, n_clusters) );
	OK_CHECK_ERR( err,
		cluster_aid_alloc(&w->h, n_vectors, n_clusters,
			CblasRowMajor) );
	if (err)
		OK_MAX_ERR( err, kmeans_work_free(w));
	return err;
}

ok_status kmeans_work_free(kmeans_work * w)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(w);

	OK_MAX_ERR( err, matrix_free(&w->A) );
	OK_MAX_ERR( err, matrix_free(&w->C) );
	OK_MAX_ERR( err, upsamplingvec_free(&w->a2c) );
	OK_MAX_ERR( err, vector_free(&w->counts) );
	OK_MAX_ERR( err, cluster_aid_free(&w->h) );
	memset(w, 0, sizeof(*w));
	return err;
}

ok_status kmeans_work_subselect(kmeans_work * w, size_t n_subvectors,
	size_t n_subclusters, size_t subvec_length)
{
	OK_CHECK_PTR(w);

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
	if (!w || !A || !C || !a2c || !counts)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	OK_RETURNIF_ERR( matrix_memcpy_ma(&w->A, A, orderA) );
	OK_RETURNIF_ERR( matrix_memcpy_ma(&w->C, C, orderC) );
	OK_RETURNIF_ERR( indvector_memcpy_va(&w->a2c.vec, a2c, stride_a2c) );
	return vector_memcpy_va(&w->counts, counts, stride_counts);
}

ok_status kmeans_work_extract(ok_float * C,
	const enum CBLAS_ORDER orderC, size_t * a2c, size_t stride_a2c,
	ok_float * counts, size_t stride_counts, kmeans_work * w)
{
	if (!w || !C || !a2c || !counts)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	OK_RETURNIF_ERR( matrix_memcpy_am(C, &w->C, orderC) );
	OK_RETURNIF_ERR( indvector_memcpy_av(a2c, &w->a2c.vec, stride_a2c) );
	return vector_memcpy_av(counts, &w->counts, stride_counts);
}

ok_status cluster(matrix * A, matrix * C, upsamplingvec * a2c,
	cluster_aid * helper, ok_float maxdist)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_PTR(helper);
	cluster_aid * h = helper;

	/*
	 * Prep work: set
	 *	h->c_squared_k = c_k'c_k,
	 *	D_ki = - 2 * c_k'a_i
	 */
	OK_RETURNIF_ERR(
		linalg_matrix_row_squares(CblasNoTrans, C, &h->c_squared) );
	OK_RETURNIF_ERR(
		blas_gemm(h->hdl, CblasNoTrans, CblasTrans, -2 * kOne,
		C, A, kZero, &h->D) );

	/* Form D_{ki} = - 2 * c_k'a_i + c_k^2 */
	OK_RETURNIF_ERR(
		linalg_matrix_broadcast_vector(&h->D, &h->c_squared,
			OkTransformAdd, CblasLeft) );

	/* set tentative cluster assigment of vector i argmin_k {D_ki} */
	OK_RETURNIF_ERR(
		linalg_matrix_reduce_indmin(&h->a2c_tentative.vec, &h->d_min,
			&h->D, CblasLeft) );

	/* finalize cluster assignements */
	// if (maxdist == OK_INFINITY)
		return OK_SCAN_ERR(
			assign_clusters_l2(A, C, a2c, h) );
	// else
		// return OK_SCAN_ERR(
			// assign_clusters_l2_lInf_cap(A, C, a2c, h, maxdist) );
}

/*
 * given matrices A and C, and upsampling linear operator U, set
 *
 *	C = diag(U'1)^{-1} * U'A
 */
ok_status calculate_centroids(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts, cluster_aid * helper)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_VECTOR(counts);
	OK_CHECK_PTR(helper);

	OK_RETURNIF_ERR( upsamplingvec_mul_matrix(helper->hdl, CblasTrans,
		CblasNoTrans, CblasNoTrans, kOne, a2c, A, kZero, C) );
	OK_RETURNIF_ERR( upsamplingvec_count(a2c, counts) );
	OK_RETURNIF_ERR( vector_safe_recip(counts) );
	return linalg_matrix_broadcast_vector(C, counts, OkTransformScale,
		CblasLeft);
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
		*tol = (kOne + ((ok_float)(*iter) / (ok_float)(*maxiter)) *
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
	int valid;

	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_VECTOR(counts);

	/* bounds/dimension checks */
	OK_RETURNIF_ERR( upsamplingvec_check_bounds(a2c) );
	valid = (counts->size == C->size1) && (A->size2 == C->size2);
	valid &= (A->size1 == a2c->size1) && (C->size1 >= a2c->size2);
	valid &= (h->D.size1 == C->size1) && (h->D.size2 == A->size1);
	if (!valid)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	/* calculate max entry of A */
	for (iter = 0; iter < A->size1 && !err; ++iter) {
		matrix_row(&vA, A, iter);
		vector_max(&vA, &rowmax);
		maxA = maxA > rowmax ? maxA : rowmax;
	}

	/* ensure C is initialized */
	OK_CHECK_ERR( err, calculate_centroids(A, C, a2c, counts, h) );

	if (!err && verbose)
		printf("\nstarting k-means on %zu vectors and %zu centroids\n",
			A->size1, C->size1);

	for (iter = 0; iter < maxiter && !err; ++iter) {
		get_distance_tolerance(&tol, &maxA, &dist_reltol, &iter,
			&maxiter);
		printf("%s %zu %s %f\n", "ITER", iter, "DISTANCE TOLERANCE",
			tol);
		OK_CHECK_ERR( err, cluster(A, C, a2c, h, tol) );
		OK_CHECK_ERR( err, calculate_centroids(A, C, a2c, counts, h) );
		if (verbose)
			printf(iterfmt, iter, h->reassigned, change_abstol);
		if (h->reassigned < change_abstol)
			break;
	}
	if (!err && verbose)
		printf(exitfmt, iter + 1, err);

	return err;
}

void * kmeans_easy_init(size_t n_vectors, size_t n_clusters, size_t vec_length)
{
	kmeans_work * w = OK_NULL;
	#ifdef __CUDACC__
	w = (kmeans_work *) malloc(sizeof(*w));
	memset(w, 0, sizeof(*w));
	#else
	ok_alloc(w, sizeof(*w));
	#endif
	kmeans_work_alloc(w, n_vectors, n_clusters, vec_length);
	return (void * ) w;
}

ok_status kmeans_easy_resize(const void * work, size_t n_vectors,
	size_t n_clusters, size_t vec_length)
{
	OK_CHECK_PTR(work);
	return kmeans_work_subselect((kmeans_work *) work, n_vectors,
		n_clusters, vec_length);
}

ok_status kmeans_easy_run(const void * work, const kmeans_settings * const s,
	const kmeans_io * io)
{
	if (!work || !s || !io)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	kmeans_work * w = (kmeans_work *) work;

	OK_RETURNIF_ERR( kmeans_work_load(w, io->A, io->orderA, io->C,
		io->orderC, io->a2c, io->stride_a2c, io->counts,
		io->stride_counts) );

	OK_RETURNIF_ERR( k_means(&w->A, &w->C, &w->a2c, &w->counts,
		&w->h, s->dist_reltol, s->change_abstol, s->maxiter,
		s->verbose) );

	return kmeans_work_extract(io->C, io->orderC,
		io->a2c, io->stride_a2c, io->counts, io->stride_counts, w);
}
ok_status kmeans_easy_finish(void * work)
{
	ok_status err = kmeans_work_free((kmeans_work *) work);
	if (work)
		ok_free(work);
	return err;
}

#ifdef __cplusplus
}
#endif
