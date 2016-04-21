	#include "optkit_defs_gpu.h"
#include "optkit_upsampling_vector.h"

inline __device__ ok_float& __get(ok_float * data, uint i, uint j,
	const uint stride_row, const uint stride_col)
{
	return data[i * stride_row + j * stride_col];
}

#ifdef __cplusplus
extern "C" {
#endif

static inline int upsampling_dims_compatible(const enum CBLAS_TRANSPOSE tU,
	const upsamplingvec * u, const size_t input_dim1,
	const size_t input_dim2, const size_t output_dim1,
	const size_t output_dim2)
{
	if (tU == CblasTrans)
		return (input_dim2 != output_dim2) &&
			((u->size2 <= output_dim1) && (u->size1 == input_dim1));
	else
		return (input_dim2 != output_dim2) &&
			((u->size1 == output_dim1) && (u->size2 <= input_dim1));
}

ok_status upsamplingvec_alloc(upsamplingvec * u, size_t size1, size_t size2)
{
	if (!u)
		return OPTKIT_ERROR_UNALLOCATED;
	else if (u->indices)
		return OPTKIT_ERROR_OVERWRITE;
	memset(u, 0, sizeof(*u));
	u->size1 = size1;
	u->size2 = size2;
	indvector_calloc(&u->vec, size1);
	u->indices = u->vec.data;
	u->stride = u->vec.stride;
	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_free(upsamplingvec * u)
{
	if (!u || !(u->indices))
		return OPTKIT_ERROR_UNALLOCATED;
	ok_free(u->vec.data);
	memset(u, 0, sizeof(*u));
	return OPTKIT_SUCCESS;
}

ok_status upsamplingvec_check_bounds(const upsamplingvec * u)
{
	if (indvector_max(&u->vec) < u->size2)
		return OPTKIT_SUCCESS;
	else
		return OPTKIT_ERROR_DIMENSION_MISMATCH;
}

ok_status upsamplingvec_update_size(upsamplingvec * u)
{
	if (!u || !u->indices)
		return OPTKIT_ERROR_UNALLOCATED;

	u->size2 = indvector_max(&u->vec);
	return OPTKIT_SUCCESS;
}


ok_status upsamplingvec_subvector(upsamplingvec * usub, upsamplingvec * u,
	size_t offset1, size_t length1, size_t size2)
{
	indvector_subvector(&usub->vec, &u->vec, offset1, length1);
	usub->indices = usub->vec.data;
	usub->size1 = length1;
	usub->size2 = size2;
	return OPTKIT_SUCCESS;
}

static __global__ void __uvec_mul_matrix(const ok_float alpha,
	ok_float * M_in, size_t * uvec, ok_float * M_out,
	const uint stride_row_in, const uint stride_col_in,
	const uint stride_u, const uint stride_row_out,
	const uint stride_col_out, const uint size1_out, const uint size2_out)
{
	uint i, j;
	uint thread_id_row = blockIdx.x * blockDim.x + threadIdx.x;
	uint thread_id_col = blockIdx.y * blockDim.y + threadIdx.y;
	uint incr_x = gridDim.x * blockDim.x;
	uint incr_y = gridDim.y * blockDim.y;
	for (i = thread_id_row; i < size1_out; i += incr_x)
		for (j = thread_id_col; j < size2_out; j += incr_y)
			__get(M_out, i, j, stride_row_out, stride_col_out) +=
				alpha * __get(M_in, uvec[i * stride_u], j,
					stride_row_in, stride_col_in);
}

static __global__ void __uvec_T_mul_matrix(const ok_float alpha,
	ok_float * M_in, size_t * uvec, ok_float * M_out,
	const uint stride_row_in, const uint stride_col_in, const uint stride_u,
	const uint stride_row_out, const uint stride_col_out,
	const uint size1_in, const uint size2_in)
{
	uint i, j;
	uint thread_id_row = blockIdx.x * blockDim.x + threadIdx.x;
	uint thread_id_col = blockIdx.y * blockDim.y + threadIdx.y;
	uint incr_x = gridDim.x * blockDim.x;
	uint incr_y = gridDim.y * blockDim.y;
	for (i = thread_id_row; i < size1_in; i += incr_x)
		for (j = thread_id_col; j < size2_in; j += incr_y)
			__get(M_out, i, j, stride_row_out, stride_col_out) +=
				alpha * __get(M_in, uvec[i * stride_u], j,
					stride_row_in, stride_col_in);
}

ok_status upsamplingvec_mul_matrix(const enum CBLAS_TRANSPOSE transU,
	const enum CBLAS_TRANSPOSE transI, const enum CBLAS_TRANSPOSE transO,
	const ok_float alpha, upsamplingvec * u, matrix * M_in, ok_float beta,
	matrix * M_out)
{
	size_t dim_in1 = (transI == CblasNoTrans) ? M_in->size1 : M_in->size2;
	size_t dim_in2 = (transI == CblasNoTrans) ? M_in->size2 : M_in->size1;
	size_t dim_out1 = (transO == CblasNoTrans) ? M_out->size1 : M_out->size2;
	size_t dim_out2 = (transO == CblasNoTrans) ? M_out->size2 : M_out->size1;
	uint row_stride_in, row_stride_out, col_stride_in, col_stride_out;

	uint grid_dim_x = (transU == CblasNoTrans) ? calc_grid_dim(dim_out1) :
		calc_grid_dim(dim_in1);
	uint grid_dim_y = (transU == CblasNoTrans) ? calc_grid_dim(dim_out2) :
		calc_grid_dim(dim_in2);
	dim3 grid_dim(grid_dim_x, grid_dim_y, 1u);
	dim3 block_dim(kBlockSize2D, kBlockSize2D, 1u);

	if ((!u || !M_in || !M_out) ||
	    (!u->indices || !M_in->data ||!M_out->data))
		return OPTKIT_ERROR_UNALLOCATED;
	if (!upsampling_dims_compatible(transU, u, dim_in1, dim_in2,
		dim_out1, dim_out2))
		return OPTKIT_ERROR_DIMENSION_MISMATCH;

	row_stride_in =
		((transI == CblasNoTrans) == (M_in->order == CblasRowMajor)) ?
		1 : (uint) M_in->ld;
	row_stride_out =
		((transO == CblasNoTrans) == (M_out->order == CblasRowMajor)) ?
		1 : (uint) M_out->ld;
	col_stride_in = (row_stride_in == 1) ? (uint) M_in->ld : 1;
	col_stride_out = (row_stride_out == 1) ? (uint) M_out->ld : 1;

	matrix_scale(M_out, beta);

	if (transU == CblasNoTrans)
		__uvec_mul_matrix<<<grid_dim, block_dim>>>(alpha, M_in->data,
			u->indices, M_out->data, row_stride_in, col_stride_in,
			u->stride, row_stride_out, col_stride_out, M_out->size1,
			M_out->size2);

	else
		__uvec_T_mul_matrix<<<grid_dim, block_dim>>>(alpha, M_in->data,
			u->indices, M_out->data, row_stride_in, col_stride_in,
			u->stride, row_stride_out, col_stride_out, M_in->size1,
			M_in->size2);

	cudaDeviceSynchronize();
	CUDA_CHECK_ERR;

	return OPTKIT_SUCCESS;
}

static __global__ void __upsampling_count(size_t * indices,
	ok_float * counts, size_t stride_idx, size_t stride_cts, size_t size)
{
	size_t i;
	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
		i += gridDim.x * blockDim.x)
		counts[indices[i * stride_idx] * stride_cts] += kOne;
}

ok_status upsamplingvec_count(const upsamplingvec * u, vector * counts)
{
	uint grid_dim = calc_grid_dim(u->size1);

	if ((!u || !counts) || (!u->indices || !counts->data))
		return OPTKIT_ERROR_UNALLOCATED;

	if (u->size2 > counts->size)
		return OPTKIT_ERROR_DIMENSION_MISMATCH;

	vector_scale(counts, kZero);

	__upsampling_count<<<grid_dim, kBlockSize>>>(u->indices, counts->data,
		u->stride, counts->stride, u->size1);

	cudaDeviceSynchronize();
	CUDA_CHECK_ERR;

	return OPTKIT_SUCCESS;
}

static __global__ void __upsampling_shift_down(size_t * data, size_t shift,
	size_t stride, size_t size)
{
	size_t i;
	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
		i += gridDim.x * blockDim.x)
		data[i * stride] -= shift;
}

static __global__ void __upsampling_shift_up(size_t * data, size_t shift,
	size_t stride, size_t size)
{
	size_t i;
	for (i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
		i += gridDim.x * blockDim.x)
		data[i * stride] += shift;
}

#ifdef __cplusplus
}
#endif
