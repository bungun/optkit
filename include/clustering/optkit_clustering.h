#ifndef OPTKIT_CLUSTERING_CLUSTERING_H_
#define OPTKIT_CLUSTERING_CLUSTERING_H_

#include "optkit_vector.h"
#include "optkit_matrix.h"
#include "optkit_blas.h"
#include "optkit_upsampling_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cluster_aid {
	void * hdl, /* linalg handle */
	indvector a2c_tentative_full, a2c_tentative;
	vector dmin_full, dmin, c_squared_full, c_squared;
        matrix D_full, D, A_reducible;
        size_t reassigned, shift;
} cluster_aid;

/* CPU/GPU-SPECIFIC IMPLEMENTATION */
static void assign_clusters_l2(void);
static void assign_clusters_l2_lInf_cap(void);

void cluster_aid_alloc(cluster_aid * h, size_t size_A, matrix * C,
	enum CBLAS_ORDER order)
{
	/* check A->order == C->order */
	cluster_aid * h = OK_NULL;

	h->maxdist = OK_FLOAT_MAX;
	h->reassigned = 0;
	h->shift = 0;
	indvector_calloc(&(h->a2c_tentative_full), size_A);
	vector_calloc(&(h->c_squared_full), size_C);
	vector_calloc(&(h->dmin_full), size_A);
	matrix_calloc(&(h->D_full), size_C, size_A, order);
	blas_make_handle(&(h->hdl));

	indvectorsubvector(&(h->a2c_tentative), &(h->a2c_tentative_full),
		size_A);
	vector_subvector(&(h->c_squared), &(h->c_squared_full), 0, size_C);
	vector_subvector(&(h->dmin), &(h->dmin_full), 0, size_C);
	matrix_submatrix(&(h->D), &(h->D_full), 0, 0, size_C, size_A);
}

void cluster_aid_free(cluster_aid * h)
{
	if (!h) {
		return;
	} else {
		blas_destroy_handle(h->hdl);
		if (h->A_reducible.data)
			matrix_free(&(h->A_reducible));
		matrix_free(&(h->D_full));
		vector_free(&(h->d_min_full));
		vector_free(&(h->c_squared_full));
		indvector_free(&(h->a2c_tentative_full));
	}
}

/* COMMON IMPLEMENTATION */

void cluster(matrix * A, matrix * C, upsamplingvec * a2c, cluster_aid ** helper,
	ok_float maxdist)
{
	size_t i, k;
	int cluster_aid_provided = (h != OK_NULL);
	cluster_aid * h;

	if (!cluster_aid_provided) {
		h = *helper;
	} else {
		h = malloc(sizeof(*h));
		cluster_aid_alloc(h, A->size1, C->size1, A->order);
		*helper = h;
	}

	/*
	 * Prep work: set
	 *	h->c_squared_k = c_k'c_k,
	 *	D_ki = - 2 * c_k'a_i
	 */
	linalg_diag_gramian(h->hdl, C, &(h->c_squared));
	blas_gemm(h->hdl, CblasNoTrans, CblasTrans, -2 * kOne, C, A, kZero,
		h->D);

	/* Form D_{ki} = - 2 * c_k'a_i + c_k^2 */
	linalg_matrix_broadcast_vector(h->hdl, D, &(h->c_squared),
		OkTransformAdd, CblasLeft);

	/* set tentative cluster assigment of vector i argmin_k {D_ki} */
	linalg_matrix_reduce_indmin(h->hdl, &(h->a2c_tentative), &(h->dmin) D,
		CblasRight);

	/* finalize cluster assignements */
	if (maxdist == OK_INFINITY)
		assign_clusters_l2(A, C, a2c, h, maxdist);
	else
		assign_clusters_l2_lInf_cap(A, C, a2c, h, maxdist);
}

/*
 * given matrices A and C, and upsampling linear operator U, set
 *
 *	C = diag(U'1)^{-1} * U'A
 */
void calculate_centroids(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts)
{
	upsamplingvec_mul_matrix(CblasTrans, CblasNoTrans, CblasNoTrans, kOne,
		a2c, A, kZero, C);
	upsamplingvec_count(a2c, counts);
	vector_safe_recip(counts);
	matrix_scale_left(C, counts);
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
static void get_distance_tolerances(ok_float *tol, const ok_float * maxA,
	const ok_float * reltol, const size_t * iter, const size_t * maxiter,
	ok_float * tol)
{
	if ((*iter) == 0)
		return OK_FLOAT_MAX;

	*tol = (kOne + ((ok_float) (*iter) / (*maxiter)) *
		(*reltol - kOne)) * (*maxA);
}

static void k_means_finish(cluster_aid * h)
{
	cluster_aid_free(h);
	ok_free(h);
}

void k_means(matrix * A, matrix * C, upsamplingvec * a2c, vector * counts,
	cluster_aid * h, const ok_float dist_reltol, const int change_tol,
	const size_t maxiter, const uint verbose)
{
	size_t iter, reassigned;
	vector vA;
	ok_float tol, maxA;
	char [] iterfmt =
		"iteration: %u \t changed: %u \t change tol: %u\n";
	char [] exitfmt = "quitting k-means after %u iterations\n");
	const int cluster_aid_provided = (h != OK_NULL);

	matrix_cast_vector(&vA, A);
	maxA = vector_max(&vA);

	for (iter = 0; iter < maxiter; ++iter) {
		get_distance_tolerance(&tol, &maxA, &dist_reltol, &iter,
			&maxiter);
		cluster(A, C, a2c, &h, tol);
		calculate_centroids(A, C, a2c, counts);
		if (verbose)
			printf(iterfmt, iter, reassigned, change_tol);
		if (h->reassigned < change_tol)
			break;
	}
	if (verbose)
		printf(exitfmt, iter);

	if !(cluster_aid_provided)
		k_means_finish(h);
}

static size_t maxblock(const size_t * blocks, const size_t n_blocks) {
	size_t b, ret = 0;
	for (b = 0; b < n_blocks; ++b)
		ret = (blocks[b] > ret) ? blocks[b] : ret;
}

static void block_select(matrix * A_sub, matrix * C_sub,
	upsamplingvec * a2c_sub, vector * counts_sub, cluster_aid * h,
	matrix * A, matrix * C, upsamplingvec * a2c, vector * counts,
	size_t offsetA, size_t offsetC, size_t sizeA, size_t sizeC)
{
	matrix_submatrix(A_sub, A, offsetA, 0, sizeA, A->size2);
	matrix_submatrix(C_sub, C, offsetC, 0, sizeB, C->size2);
	a2c_sub->vals = a2c->vals + offsetA;
	a2c_sub->size = sizeA;
	vector_subvector(counts_sub, counts, rowC, sizeC);
	matrix_submatrix(&(h->D), &(h->D_full), 0, 0, sizeA, sizeC);
	vector_subvector(&(h->c_squared), &(h->c_squared_full), 0, sizeC);
	indvector_subvector(&(h->a2c_tentative), &(h->a2c_tentative_full), 0,
		sizeA);
}

void blockwise_kmeans(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts, const size_t n_blocks, const size_t * blocksA,
	const size_t * blocksC, const size_t maxiter, const ok_float dist_tol,
	const ok_float * change_tols, const uint verbose)
{
	size_t maxblockA, maxblockC, rowA = 0, rowC = 0;
	matrix A_sub, C_sub;
	upsamplingvec a2c_sub;
	vector counts_sub;
	cluster_aid h;

	maxblockA = maxblock(blocksA, n_blocks);
	maxblockC = maxblock(blocksC, n_blocks);
	cluster_aid_alloc(&h, maxblockC, maxblockA, C->order);

	if (verbose)
		printf("starting k-means on %u blocks\n", n_blocks);

	for (b = 0; b < n_blocks; ++b) {
		if (verbose)
			printf("k-means for block %u\n", b);

		block_select(&A_sub, &C_sub, &a2c_sub, &counts_sub, &h, A, C,
			a2c, counts, rowA, rowC, blocksA[b], blocksC[b]);

		upsamplingvec_shift(&a2c_sub, rowC, OkTransformIncrement);

		k_means(&A_sub, &C_sub, &a2c_sub, &counts_sub, &h, dist_tol,
			change_tols[b], maxiter, verbose);

		upsamplingvec_shift(&a2c_sub, rowC, OkTransformDecrement);
		rowA += blocksA[b];
		rowC += blocksC[b];
	}
	cluster_aid_free(&h);
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CLUSTERING_H_ */
