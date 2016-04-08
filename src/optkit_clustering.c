#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

void upsamplingvec_alloc(upsamplingvec * u, size_t size)
{
	u->vals = malloc(size * sizeof(size_t));
	u->size = size;
	u->shift = 0;
}

void upsamplingvec_calloc(upsamplingvec * u, size_t size)
{
	upsamplingvec_alloc(u);
	memset(u->vals, 0, n * sizeof(size_t))
}

void upsamplingvec_free(upsamplingvec * u)
{
	if (!u)
		return;
	else if (u->vals)
		ok_free(u->vals);
}

void upsamplingvec_subvector(upsamplingvec * usub, upsamplingvec * u,
	size_t offset, size_t size, size_t shift)
{
	usub->vals = u->vals + offset;
	usub->size = size;
	usub->shift = u->shift + shift;
}

void cluster_aid_init(matrix * A, matrix * C)
{
	/* check A->order == C->order */
	cluster_aid * h = OK_NULL;

	h = malloc(sizeof(*h));
	h->c_squared
	h->maxdist = OK_FLOAT_MAX;
	h->reassigned = 0;
	h->idx_tentative = 0;
	vector_calloc(&(h->c_squared_full), C->size1);
	vector_subvector(&(h->c_squared), 0, C->size1);
	matrix_calloc(&(h->D_full), C->size1, A->size1, C->order);
	matrix_submatrix(&(h->D), &(h->D_full), 0, 0, C->size1, A->size1);
	blas_make_handle(&(h->hdl));
	h->freeable = 1;
}

void cluster_aid_finish(cluster_aid * h)
{
	if (!h) {
		return;
	} else if (h->freeable) {
		blas_destroy_handle(h->hdl);
		matrix_free(h->D_full);
		vector_free(h->c_squared_full);
		ok_free(h);
	}
}

void cluster(matrix * A, matrix * C, upsamplingvec * a2c, cluster_aid ** helper,
	ok_float maxdist)
{
	size_t i, k, reassigned = 0;
	int cluster_aid_provided = (h != OK_NULL);
	cluster_aid * h;

	if (!cluster_aid_provided)
		h = *helper;
	else
		*helper = h = cluster_aid_init(A, C);

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
	for (i = 0; i < A->size1, ++i) {
		h->idx_tentative = vector_indmin(h->d) + a2c->shift;

		/*
		 * reassign vector i to cluster k if
		 *
		 * 	||a_i - c_k||_\infty <= maxdist
		 *
		 * (no action if vector i assigned to cluster k already)
		 */
		if (h->idx_tentative == a2c->vals[i]) {
			continue;
		} else if (maxdist == OK_INF) {
			a2c->vals[i] = h->idx_tentative;
			++h->reassigned;
		} else {
			matrix_row(h->c, C, h->idx_tentative);
			vector_memcpy_vv(h->c_tentative, h->c);
			blas_axpy(h->hdl, -kOne, h->a, h->c_tentative);
			vector_abs(h->c_tentative);

			h->maxdist = vector_max(h->c_tentative);
			if (h->maxdist <= maxdist) {
				a2c->vals[i] = h->idx_tentative;
				++h->reassigned;
			}
		}
	}
}

void calculate_centroids(matrix * A, matrix * C, upsamplingvec * a2c,
	vector * counts, cluster_aid * h)
{
	size_t, i, k;

	vector_scale(counts, kZero);
	matrix_scale(C, kZero);

	for (i = 0; i < a2c.size; ++i) {
		k = a2c->vals[i] - a2c->shift;
		matrix_row(h->a, A, i);
		matrix_row(h->c, C, a2c->vals[i] - a2c->shift);
		blas_axpy(h->hdl, kOne, h->a, h->c);
		counts[k] += kOne;
	}

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
		calculate_centroids(A, C, a2c, counts, h);
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
	matrix_alloc(&(h->D_full), maxblockC, maxblockA, C->order);
 	vector_alloc(&(h->c_squared_full), C->size1);

	if (verbose)
		printf("starting k-means on %u blocks\n", n_blocks);

	for (b = 0; b < n_blocks; ++b) {
		if (verbose)
			printf("k-means for block %u\n", b);

		matrix_submatrix(&A_sub, A, rowA, 0, blocksA[b], A->size2);
		matrix_submatrix(&C_sub, C, rowC, 0, blocksC[b], C->size2);
		a2c_sub.vals = a2c.vals + rowA;
		a2c_sub.size = blocksA[b];
		a2c_sub.shift += rowC;
		vector_subvector(&counts_sub, counts, rowC, blocksC[b]);
		matrix_submatrix(&(h->D), &(h->D_full), 0, 0, blocksA[b],
			blocksC[b]);
		vector_subvector(&(h->c_squared), &(h->c_squared_full), 0,
			blocksC[b]);

		k_means(A_sub, C_sub, a2c_sub, counts_sub, h, dist_tol,
			change_tols[b], maxiter, verbose);
		a2c_sub.shift -= rowC;

		rowA += blocksA[b];
		rowC += blocksC[b];
	}
	matrix_free(&(h->D_full));
}

#ifdef __cplusplus
}
#endif
