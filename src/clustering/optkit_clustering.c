#include "optkit_clustering.h"

#ifdef __cplusplus
extern "C" {
#endif

const size_t kBlockSize = 128;

/*
 * given tentative cluster assigments {(i, k)} stored in h, reassign
 * vector i to cluster k (no action if vector i assigned to cluster
 * k already).
 *
 * tally the number of reassignments.
 */
ok_status assign_clusters_l2(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_PTR(h);

	size_t i;
	upsamplingvec * u = &h->a2c_tentative;

	if (A->size1 != a2c->size1 || a2c->size2 > C->size1 ||
		A->size2 != C->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	h->reassigned = 0;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < A->size1; ++i)
		if (a2c->indices[i] != u->indices[i]) {
			a2c->indices[i] = u->indices[i];
			++h->reassigned;
		}
	return OPTKIT_SUCCESS;
}

static ok_float __dist_lInf_A_minus_UC_i(const ok_float * A,
	const int * strideA, const size_t * iterA, const upsamplingvec * u,
	const ok_float * C, const int * strideC, const size_t * iterC,
	ok_float * A_blk, const size_t * strideBlk, const size_t * strideIter,
	const size_t * block, const size_t * i, const int * row_length)
{
	ok_float dist;
	size_t idx;
	CBLAS(copy)(*row_length, A + (*i) * (*iterA), (*strideA),
		A_blk + (*i - *block) * (*strideIter),
		(const int) (*strideBlk));
	CBLAS(axpy)(*row_length, -kOne,
		C + u->indices[(*i) * u->stride] * (*iterC), (*strideC),
		A_blk + (*i - *block) * (*strideIter),
		(const int) (*strideBlk));
	idx = (size_t) CBLASI(amax)(*row_length,
		A_blk + (*i - *block) * (*strideIter),
		(const int) (*strideBlk));
	dist = A_blk[(*i - *block) * (*strideIter) + idx * (*strideBlk)];
	return MATH(fabs)(dist);
}

/*
 * given tentative cluster assigments {(i, k)} stored in h, reassign
 * vector i to cluster k if
 *
 * 	||a_i - c_k||_\infty <= maxdist
 *
 * (no action if vector i assigned to cluster k already)
 *
 * tally the number of reassignments.
 *
 */
ok_status assign_clusters_l2_lInf_cap(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h, ok_float maxdist)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_PTR(h);
	if (A->size1 != a2c->size1 || a2c->size2 > C->size1 ||
		A->size2 != C->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	size_t i, block, row_stride, idx_stride, row_strideA, row_strideC;
	int strideA, strideC;
	matrix * A_blk;
	upsamplingvec * u = &h->a2c_tentative;

	if (!h->A_reducible.data)
		err = matrix_alloc(&h->A_reducible, kBlockSize, A->size2,
			A->order);

	A_blk = &h->A_reducible;
	row_stride = (A_blk->order == CblasRowMajor) ? A_blk->ld : 1;
	idx_stride = (A_blk->order == CblasRowMajor) ? 1 : A_blk->ld;

	row_strideA = (A->order == CblasRowMajor) ? A->ld : 1;
	strideA = (A->order == CblasRowMajor) ? 1 : (int) A->ld;
	row_strideC = (C->order == CblasRowMajor) ? C->ld : 1;
	strideC = (C->order == CblasRowMajor) ? 1 : (int) C->ld;

	h->reassigned = 0;
	for (block = 0; block < A->size1; block += kBlockSize)
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (i = block; i < block + kBlockSize && i < A->size1; ++i) {
			if (a2c->indices[i] == u->indices[i]) {
				continue;
			} else if (maxdist >= __dist_lInf_A_minus_UC_i(A->data,
				&strideA, &row_strideA, u, C->data, &strideC,
				&row_strideC, A_blk->data, &idx_stride,
				&row_stride, &block, &i,
				(const int *) &A->size2)) {

				a2c->indices[i] = u->indices[i];
				++h->reassigned;
			}
		}
	return err;
}

#ifdef __cplusplus
}
#endif
