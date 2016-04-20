#ifndef OPTKIT_CLUSTERING_CLUSTERING_H_
#define OPTKIT_CLUSTERING_CLUSTERING_H_

#include "optkit_dense.h"
#include "optkit_upsampling_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cluster_aid {
	void * hdl; /* linalg handle */
	upsamplingvec a2c_tentative_full, a2c_tentative;
	vector d_min_full, d_min, c_squared_full, c_squared;
        matrix D_full, D, A_reducible;
        size_t reassigned;
} cluster_aid;

/* CPU/GPU-SPECIFIC IMPLEMENTATION */
static ok_status assign_clusters_l2(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h);
static ok_status assign_clusters_l2_lInf_cap(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h, ok_float maxdist);

ok_status cluster_aid_alloc(cluster_aid * h, size_t size_A, size_t size_C,
	enum CBLAS_ORDER order)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!h)
		return OPTKIT_ERROR_UNALLOCATED;

	h->reassigned = 0;
	err = upsamplingvec_alloc(&(h->a2c_tentative_full), size_A, size_C);
	if (!err)
		vector_calloc(&(h->c_squared_full), size_C);
		vector_calloc(&(h->d_min_full), size_A);
		matrix_calloc(&(h->D_full), size_C, size_A, order);
		err = blas_make_handle(&(h->hdl));

	if (!err)
		upsamplingvec_subvector(&(h->a2c_tentative),
			&(h->a2c_tentative_full), 0, 0, size_A, size_C);
		vector_subvector(&(h->c_squared), &(h->c_squared_full), 0,
			size_C);
		vector_subvector(&(h->d_min), &(h->d_min_full), 0, size_A);
		matrix_submatrix(&(h->D), &(h->D_full), 0, 0, size_C, size_A);
	return err;
}

ok_status cluster_aid_free(cluster_aid * h)
{
	if (!h)
		return OPTKIT_ERROR_UNALLOCATED;

	blas_destroy_handle(h->hdl);
	if (h->A_reducible.data)
		matrix_free(&(h->A_reducible));
	matrix_free(&(h->D_full));
	vector_free(&(h->d_min_full));
	vector_free(&(h->c_squared_full));
	upsamplingvec_free(&(h->a2c_tentative_full));
	return OPTKIT_SUCCESS;
}

/* COMMON IMPLEMENTATION */
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
		memset(h, 0, sizeof(*h));
		err = cluster_aid_alloc(h, A->size1, C->size1, A->order);
		if (err)
			return err;
		*helper = h;
	}

	/*
	 * Prep work: set
	 *	h->c_squared_k = c_k'c_k,
	 *	D_ki = - 2 * c_k'a_i
	 */
	linalg_matrix_row_squares(CblasNoTrans, C, &(h->c_squared));
	blas_gemm(h->hdl, CblasNoTrans, CblasTrans, -2 * kOne, C, A, kZero,
		&(h->D));

	/* Form D_{ki} = - 2 * c_k'a_i + c_k^2 */
	linalg_matrix_broadcast_vector(&(h->D), &(h->c_squared),
		OkTransformAdd, CblasLeft);

	/* set tentative cluster assigment of vector i argmin_k {D_ki} */
	linalg_matrix_reduce_indmin(&(h->a2c_tentative.vec), &(h->d_min),
		&(h->D), CblasLeft);

	/* finalize cluster assignements */
	if (maxdist == OK_INFINITY)
		err = assign_clusters_l2(A, C, a2c, h);
	else
		err = assign_clusters_l2_lInf_cap(A, C, a2c, h, maxdist);

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
	err = upsamplingvec_mul_matrix(CblasTrans, CblasNoTrans, CblasNoTrans,
		kOne, a2c, A, kZero, C);
	err = upsamplingvec_count(a2c, counts);
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

	*tol = (kOne + ((ok_float) (*iter) / (*maxiter)) *
		(*reltol - kOne)) * (*maxA);
}

static ok_status k_means_finish(cluster_aid * h)
{
	ok_status err = cluster_aid_free(h);
	ok_free(h);
	return err;
}

ok_status k_means(matrix * A, matrix * C, upsamplingvec * a2c, vector * counts,
	cluster_aid * h, const ok_float dist_reltol, const size_t change_tol,
	const size_t maxiter, const uint verbose)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t iter;
	vector vA;
	ok_float tol, maxA;
	char iterfmt[] =
		"iteration: %u \t changed: %u \t change tol: %u\n";
	char exitfmt[] = "quitting k-means after %u iterations, error = %u\n";
	const int cluster_aid_provided = (h != OK_NULL);
	int valid;

	valid = (counts->size == C->size1) && (A->size2 == C->size2);
	valid &= (A->size1 == a2c->size1) && (C->size2 >= a2c->size2);
	if (!valid)
		return OPTKIT_ERROR_DIMENSION_MISMATCH;

	matrix_cast_vector(&vA, A);
	maxA = vector_max(&vA);

	if (verbose)
		printf("\nstarting k-means on %zu vectors and %zu centroids\n",
			A->size1, C->size1);

	for (iter = 0; iter < maxiter && !err; ++iter) {
		get_distance_tolerance(&tol, &maxA, &dist_reltol, &iter,
			&maxiter);
		err = cluster(A, C, a2c, &h, tol);
		if (!err)
			err = calculate_centroids(A, C, a2c, counts);
		if (verbose)
			printf(iterfmt, iter, h->reassigned, change_tol);
		if (h->reassigned < change_tol)
			break;
	}
	if (verbose)
		printf(exitfmt, iter + 1, err);

	if (!cluster_aid_provided)
		err = k_means_finish(h);

	return err;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CLUSTERING_H_ */
