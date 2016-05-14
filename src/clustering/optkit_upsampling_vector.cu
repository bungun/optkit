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

ok_status upsamplingvec_mul_matrix(void * linalg_handle,
	const enum CBLAS_TRANSPOSE transU, const enum CBLAS_TRANSPOSE transI,
	const enum CBLAS_TRANSPOSE transO, const ok_float alpha,
	upsamplingvec * u, matrix * M_in, ok_float beta, matrix * M_out)
{
	OK_CHECK_UPSAMPLINGVEC(u);
	OK_CHECK_MATRIX(M_in);
	OK_CHECK_MATRIX(M_out);

	ok_status err = OPTKIT_SUCCESS:
	size_t i, dim_in1, dim_in2, dim_out1, dim_out2;
	size_t ptr_stride_in, ptr_stride_out;
	int stride_in, stride_out;
	const int transpose = transU == CblasTrans;

	if ((!u || !M_in || !M_out) ||
	    (!u->indices || !M_in->data ||!M_out->data))
		return OPTKIT_ERROR_UNALLOCATED;

	dim_in1 = (transI == CblasNoTrans) ? M_in->size1 : M_in->size2;
	dim_in2 = (transI == CblasNoTrans) ? M_in->size2 : M_in->size1;
	dim_out1 = (transO == CblasNoTrans) ? M_out->size1 : M_out->size2;
	dim_out2 = (transO == CblasNoTrans) ? M_out->size2 : M_out->size1;

	if (!upsampling_dims_compatible(transpose, u, dim_in1, dim_in2,
		dim_out1, dim_out2))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	stride_in =
		((transI == CblasNoTrans) == (M_in->order == CblasRowMajor)) ?
		1 : (int) M_in->ld;
	stride_out =
		((transO == CblasNoTrans) == (M_out->order == CblasRowMajor)) ?
		1 : (int) M_out->ld;
	ptr_stride_in = (stride_in == 1) ? M_in->ld : 1;
	ptr_stride_out = (stride_out == 1) ? M_out->ld : 1;

	OK_RETURNIF_ERR( matrix_scale(M_out, beta) );
	OK_RETURNIF_ERR( matrix_scale(M_out, beta) );

	if (!transpose)
		for (i = 0; i < dim_out1 && !err; ++i) {
			size_t row_;
			cudaStream_t s;
			OK_CHECK_CUDA( err,
				cudaStreamCreate(&s) );
			OK_CHECK_CUDA( err,
				cudaMemcpyAsync(&row_,
					u->indices + i * u->stride,
					sizeof(row_), cudaMemcpyDeviceToHost,
					s) );
			OK_CHECK_CUBLAS( err,
				cublasSetStream(
					*(cublasHandle_t *) linalg_handle,
					s) );
			OK_CHECK_CUBLAS(
				CUBLAS(axpy)(*(cublasHandle_t *) linalg_handle,
					dim_in2, alpha,
					M_in->data + row_ * ptr_stride_in,
					stride_in,
					M_out->data + i * ptr_stride_out,
					stride_out) );
			OK_CHECK_CUDA( err,
				cudaStreamDestroy(s) );
		}
	else
		for (i = 0; i < dim_in1; ++i) {
			size_t row_;
			cudaStream_t s;
			OK_CHECK_CUDA( err,
				cudaStreamCreate(&s) );
			OK_CHECK_CUDA( err,
				cudaMemcpyAsync(&row_,
					u->indices + i * u->stride,
					sizeof(row_), cudaMemcpyDeviceToHost,
					s) );
			OK_CHECK_CUBLAS( err,
				cublasSetStream(
					*(cublasHandle_t *) linalg_handle,
					s) );
			OK_CHECK_CUBLAS( err,
				CUBLAS(axpy)(*(cublasHandle_t *) linalg_handle,
					dim_in2, alpha,
					M_in->data + i * ptr_stride_in,
					stride_in,
					M_out->data + row_ * ptr_stride_out,
					stride_out) );
			OK_CHECK_CUDA( err,
				cudaStreamDestroy(s) );
		}

	cudaDeviceSynchronize();
	return OK_MAX_ERR( err, OK_CHECK_CUDA );
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
	if ((!u || !counts) || (!u->indices || !counts->data))
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	uint grid_dim = calc_grid_dim(u->size1);

	if (u->size2 > counts->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_RETURNIF_ERR( vector_scale(counts, kZero) );

	__upsampling_count<<<grid_dim, kBlockSize>>>(u->indices, counts->data,
		u->stride, counts->stride, u->size1);

	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

#ifdef __cplusplus
}
#endif
