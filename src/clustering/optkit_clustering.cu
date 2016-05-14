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
	size_t row = i + blockIdx.x * blockDim.x + threadIdx.x;
	while (row < len_a2c) {
		new_ = a2c_tentative + row * stride_tentative;
		curr = a2c_curr + row * stride_curr;
		*reassigned += (size_t)(*curr != *new_);
		*curr = *new_;
		row += blockDim.x * gridDim.x;
	}
}

ok_status assign_clusters_l2(matrix * A, matrix * C, upsamplingvec * a2c,
	cluster_aid * h)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t i, * reassigned;
	uint grid_dim;

	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_PTR(h);
	if (A->size1 != a2c->size1 || a2c->size2 > C->size1 ||
		A->size2 != C->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	grid_dim = calc_grid_dim(A->size1);
	h->reassigned = 0;
	OK_CHECK_ERR( err,
		ok_alloc_gpu(reassigned, sizeof(size_t)) );
	OK_CHECK_ERR( err,
		ok_memcpy_gpu(reassigned, &h->reassigned, sizeof(size_t)) );

	for (i = 0; i < A->size1 && !err; i += grid_dim * kBlockSize) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		__assign_clusters_l2<<<grid_dim, kBlockSize, 0, s>>>(
			a2c->indices, a2c->size1, a2c->stride,
			h->a2c_tentative.indices, h->a2c_tentative.stride,
			reassigned, i);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	err = OK_STATUS_CUDA;

	OK_CHECK_ERR( err,
		ok_memcpy_gpu(&h->reassigned, reassigned, sizeof(size_t)) );
	return err;
}

namespace optkit {
static __device__ ok_float reduce_amuc_max(ok_float & val, const ok_float & a,
	const ok_float & c, const ok_float & maxdist)
{
	val = MATH(fmax)(val, MATH(fabs)(a - c));
	return (ok_float) val > maxdist;
}
} /* namespace optkit */

static __global__ void __dist_lInf_a_minus_uc(ok_float * const a,
	const size_t stride_a, size_t * const u, const size_t stride_u,
	ok_float * const C, const size_t rowstride_c, const size_t colstride_c,
	const size_t ncols, const ok_float maxdist, ok_float * max_violated)
{
	ok_float * c = C + u[stride_u] * rowstride_c;
	uint col = threadIdx.x;
	uint block_stride = 2 * kBlockSize;
	uint global_col = blockIdx.x * block_stride + col;
	uint gride_stride = gridDim.x * block_stride;
	ok_float mv = *max_violated;
	__shared__ ok_float a_minus_c[kBlockSize];

	/* copy Asub = A */
	while (global_col < ncols && mv == 0) {
		mv += optkit::reduce_amuc_max(a_minus_c[col],
			a[global_col * stride_a],
			c[global_col * colstride_c], maxdist);
		if (global_col + kBlockSize < ncols && mv == 0)
			mv += optkit::reduce_amuc_max(a_minus_c[col],
				a[(global_col + kBlockSize) * stride_a],
				c[(global_col + kBlockSize) * colstride_c],
				maxdist);
		global_col += gride_stride;
	}
	__syncthreads();
	*max_violated += mv;
  }

static __global__ void __assign_clusters_l2_lInf_cap(size_t * a2c_curr,
	const size_t len_a2c, const size_t stride_curr,
	size_t * const a2c_tentative, const size_t stride_tentative,
	ok_float * const dist_violations, const size_t stride_dv,
	size_t * reassigned, const size_t i)
{
	size_t * new_, * curr;
	size_t row = i + blockIdx.x * blockDim.x + threadIdx.x;
	while (row < len_a2c) {
		new_ = a2c_tentative + row * stride_tentative;
		curr = a2c_curr + row * stride_curr;
		if (dist_violations[row * stride_dv] == 0) {
			*reassigned += (size_t)(*curr != *new_);
			*curr = *new_;
		}
		row += blockDim.x * gridDim.x;
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
ok_status assign_clusters_l2_lInf_cap(matrix * A, matrix * C,
	upsamplingvec * a2c, cluster_aid * h, ok_float maxdist)
{
	ok_status err = OPTKIT_SUCCESS;
	uint i, grid_dim;

	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	OK_CHECK_UPSAMPLINGVEC(a2c);
	OK_CHECK_PTR(h);
	if (A->size1 != a2c->size1 || a2c->size2 > C->size1 ||
		A->size2 != C->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	size_t row_stride_A = (A->order == CblasRowMajor) ? A->ld : 1;
	size_t col_stride_A = (A->order == CblasRowMajor) ? 1 : A->ld;
	size_t row_stride_C = (C->order == CblasRowMajor) ? C->ld : 1;
	size_t col_stride_C = (C->order == CblasRowMajor) ? 1 : A->ld;
	size_t * reassigned;

	vector * dmax = &h->d_min;

	h->reassigned = 0;
	OK_CHECK_ERR( err,
		ok_alloc_gpu(reassigned, sizeof(size_t)) );
	OK_CHECK_ERR( err,
		ok_memcpy_gpu(reassigned, &h->reassigned, sizeof(size_t)) );

	/* zero out distances */
	OK_CHECK_ERR( err, vector_scale(dmax, kZero) );

	for (i = 0; i < A->size1; ++i) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		__dist_lInf_a_minus_uc<<<grid_dim, kBlockSize, 0, s>>>(
			A->data + i * row_stride_A, col_stride_A, a2c->indices,
			a2c->stride, C->data, row_stride_C, col_stride_C,
			A->size2, maxdist, dmax->data + i * dmax->stride);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	err = OK_STATUS_CUDA;

	grid_dim = calc_grid_dim(A->size1);
	for (i = 0; i < A->size1 && !err; i += grid_dim * kBlockSize) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		__assign_clusters_l2_lInf_cap<<<grid_dim, kBlockSize, 0, s>>>(
			a2c->indices, a2c->size1, a2c->stride,
			h->a2c_tentative.indices, h->a2c_tentative.stride,
			dmax->data, dmax->stride, reassigned, i);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	err = OK_STATUS_CUDA;

	OK_CHECK_ERR( err,
		ok_memcpy_gpu(&h->reassigned, reassigned, sizeof(size_t)) );
	return err;
}

#ifdef __cplusplus
}
#endif
