#include "optkit_defs_gpu.h"
#include "optkit_clustering.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * given tentative cluster assigments {(i, k)} stored in h, reassign
 * vector i to cluster k (no action if vector i assigned to cluster
 * k already).
 *
 * tally the number of reassignments.
 */
static __global__ void __assign_clusters_l2(size_t * a2c_curr,
	const size_t len_a2c, const size_t stride_curr, size_t * a2c_tentative,
	const size_t stride_tentative, size_t * reassigned, const size_t i)
{
	size_t * new_, * curr;
	size_t row = i + threadIdx.x;
	if (row < len_a2c) {
		new_ = a2c_tentative + row * stride_tentative;
		curr = a2c_curr + row * stride_curr;
		*reassigned += (size_t)(*curr != *new_);
		*curr = *new_;
	}
}

static ok_status assign_clusters_l2(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t i, * reassigned;
	h->reassigned = 0;
	ok_alloc_gpu(reassigned, sizeof(size_t));
	ok_memcpy_gpu(reassigned, &(h->reassigned), sizeof(size_t));

	for (i = 0; i < A->size1; i += kBlockSize) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		__assign_clusters_l2<<<1, kBlockSize, 0, s>>>(a2c->indices,
			a2c->size1, a2c->stride, h->a2c_tentative.indices,
			h->a2c_tentative.stride, reassigned, i);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	CUDA_CHECK_ERR;

	ok_memcpy_gpu(&(h->reassigned), reassigned, sizeof(size_t));
	return err;
}

static __global__ void __dist_lInf_A_minus_UC(const ok_float * A,
	const size_t rowsA, const size_t colsA,
	const size_t row_stride_A, const size_t col_stride_A,
	const size_t * a2c, const size_t stride_a2c, const ok_float * C,
	const size_t row_stride_C, const size_t col_stride_C,
	ok_float * distances, const size_t stride_dist,
	const size_t i, const size_t j)
{
	uint row = threadIdx.x;
	uint col = threadIdx.y;
	uint global_row = i + row;
	uint global_col = j + col;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileSize * kTileLD];
	__shared__ ok_float Csub[kTileSize * kTileLD];
	__shared__ ok_float dist_sub[kTileSize];

	if (global_row >= rowsA || global_col >= colsA)
		return;

	/* initialize distances -> 0 */
	if (col == 0)
		dist_sub[row] = 0;
	__syncthreads();

	/* copy Asub = A */
	Asub[row * kTileLD + col] = A[global_row * row_stride_A +
		global_col * col_stride_A];

	/* Csub = UC */
	Csub[row * kTileLD + col] = C[a2c[global_row * stride_a2c] *
		row_stride_A + global_col * col_stride_A];
	__syncthreads();

	/* A -= UC */
	Asub[row * kTileLD + col] -= Csub[row * kTileLD + col];
	__syncthreads();

	/* set amax */
	dist_sub[row] = MATH(fmax)(dist_sub[row],
		MATH(fabs)(Asub[row * kTileLD + col]));
	__syncthreads();

	/* copy distances */
	if (col == 0)
		distances[global_row * stride_dist] = MATH(fmax)(
			distances[global_row * stride_dist], dist_sub[row]);
}

static __global__ void __assign_clusters_l2_lInf_cap(size_t * a2c_curr,
	const size_t len_a2c, const size_t stride_curr, size_t * a2c_tentative,
	const size_t stride_tentative, const ok_float * distances,
	const size_t stride_dist, const ok_float maxdist, size_t * reassigned,
	const size_t i)
{
	uint row = i + threadIdx.x;
	if (row < len_a2c && distances[row * stride_dist] <= maxdist) {
		a2c_curr[row * stride_curr] =
			a2c_tentative[row * stride_tentative];
		++(*reassigned);
	}
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
static ok_status assign_clusters_l2_lInf_cap(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h, ok_float maxdist)
{
	ok_status err = OPTKIT_SUCCESS;
	uint i, j;
	uint block_size = kTiles2D * kTileSize;

	dim3 grid_dim(kTileSize, kTileSize);
	dim3 blk_dim(kTiles2D, kTiles2D);

	size_t row_stride_A = (A->order == CblasRowMajor) ? A->ld : 1;
	size_t col_stride_A = (A->order == CblasRowMajor) ? 1 : A->ld;
	size_t row_stride_C = (C->order == CblasRowMajor) ? C->ld : 1;
	size_t col_stride_C = (C->order == CblasRowMajor) ? 1 : A->ld;

	size_t * reassigned;

	vector * dmax = &(h->d_min);

	h->reassigned = 0;
	ok_alloc_gpu(reassigned, sizeof(size_t));
	ok_memcpy_gpu(reassigned, &(h->reassigned), sizeof(size_t));

	/* zero out distances */
	vector_scale(dmax, kZero);

	/* reduce one bundle of [block_size] rows x N cols per stream */
	for (i = 0; i < A->size1; i += block_size) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		for (j = 0; j < A->size2; j += block_size)
			__dist_lInf_A_minus_UC<<<grid_dim, blk_dim, 0, s>>>(
				A->data, A->size1, A->size2, row_stride_A,
				col_stride_A, a2c->indices, a2c->stride,
				C->data, row_stride_C, col_stride_C,
				dmax->data, dmax->stride, i, j);

		__assign_clusters_l2_lInf_cap<<<1, block_size, 0, s>>>(
			a2c->indices, a2c->size1, a2c->stride,
			h->a2c_tentative.indices, h->a2c_tentative.stride,
			dmax->data, dmax->stride, maxdist, reassigned, i);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	CUDA_CHECK_ERR;

	ok_memcpy_gpu(&(h->reassigned), reassigned, sizeof(size_t));
	return err;
}

#ifdef __cplusplus
}
#endif
