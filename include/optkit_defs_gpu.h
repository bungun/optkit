#ifndef OPTKIT_DEFS_GPU_H_
#define OPTKIT_DEFS_GPU_H_

#include "optkit_defs.h"
#include <math_constants.h>

#ifdef __cplusplus
extern "C" {
#endif

const unsigned int kTileSize = 32u;
const unsigned int kBlockSize = 1024u;
const unsigned int kBlockSize2D = 32u;
const unsigned int kTiles2D = 8u;
const unsigned int kMaxGridSize = 65535u;

#ifndef CUDA_CHECK_ERR
#define CUDA_CHECK_ERR \
	do { \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess) { \
			printf("%s:%d:%s\n ERROR_CUDA: %s\n", \
				__FILE__, __LINE__, __func__, \
				cudaGetErrorString(err)); \
		} \
	} while (0)
#endif

/*
 * http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-
 * check-for-errors-using-the-cuda-runtime-api
 */
// #define CUDA_CHECK_ERR(ans) \
// 	do { \
// 		ok_cuda_status((ans), __FILE__, __LINE__, __func__); \
// 	} while (0)

#define CUBLAS_CHECK_ERR(ans) \
	do { \
		ok_cublas_status((ans), __FILE__, __LINE__, __func__); \
	} while (0)

#define CUSPARSE_CHECK_ERR(ans) \
	do { \
		ok_cusparse_status((ans), __FILE__, __LINE__, __func__); \
	} while (0)

#define ok_alloc_gpu(x, n) \
	do { \
		cudaMalloc((void **) &x, n); \
		CUDA_CHECK_ERR; \
	} while (0)
// #define ok_alloc_gpu(x, n) OK_CHECK_CUDA( cudaMalloc((void **) &x, n) )


#define ok_memcpy_gpu(x, y, n) \
	do { \
		cudaMemcpy(x, y, n, cudaMemcpyDefault); \
		CUDA_CHECK_ERR; \
	} while (0)
// #define ok_memcpy_gpu(x, y, n) OK_CHECK_CUDA( cudaMemcpy(x, y, n, cudaMemcpyDefault); )


#define ok_free_gpu(x) \
	do { \
		cudaFree(x); \
		CUDA_CHECK_ERR; \
		x = OK_NULL; \
	} while(0)
// #define ok_free_gpu(x, y, n) do { OK_CHECK_CUDA( cudaFree(x); ); x = OK_NULL; } while(0)


#ifndef FLOAT
	#define CUBLAS(x) CUDA_CHECK_ERR; cublasD ## x
	#define CUSPARSE(x) CUDA_CHECK_ERR; cusparseD ## x
	#define OK_CUDA_NAN CUDART_NAN
#else
	#define CUBLAS(x) CUDA_CHECK_ERR; cublasS ## x
	#define CUSPARSE(x) CUDA_CHECK_ERR; cusparseS ## x
	#define OK_CUDA_NAN CUDART_NAN_F
#endif


inline uint calc_grid_dim(size_t size)
{
	return (uint) min( ((uint) size + kBlockSize - 1u) / kBlockSize,
		kMaxGridSize);
}

/*
 * status code to string conversion from POGS
 * https://github.com/foges/pogs/blob/master/src/gpu/include/cml/cml_utils.cuh
 */
static const char* cublas_err2string(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	default:
	return "<unknown>";
	}
}

static const char* cusparse_err2string(cusparseStatus_t error) {
	switch (error) {
	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";
	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";
	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";
	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";
	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";
	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";
	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";
	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";
	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	default:
		return "<unknown>";
	}
}

inline ok_status ok_cuda_status(cudaError_t code, const char *file, int line,
	const char *function)
{
	if (code != cudaSuccess) {
		printf("%s:%d:%s\n ERROR_CUDA: %s\n", file, line, function,
			cudaGetErrorString(code));
		return OPTKIT_ERROR_CUDA;
	} else {
		return OPTKIT_SUCCESS;
	}
}

inline ok_status ok_cublas_status(cublasStatus_t code, const char *file, int line,
	const char *function)
{
	if (code != CUBLAS_STATUS_SUCCESS) {
		printf("%s:%d:%s\n ERROR_CUDA: %s\n", file, line, function,
	        	cublas_err2string(code));
		return OPTKIT_ERROR_CUBLAS;
	} else {
		return OPTKIT_SUCCESS;
	}
}

inline ok_status ok_cusparse_status(cusparseStatus_t code, const char *file,
	int line, const char *function)
{
	if (code != CUSPARSE_STATUS_SUCCESS) {
		printf("%s:%d:%s\n ERROR_CUDA: %s\n", file, line, function,
			cusparse_err2string(code));
		return OPTKIT_ERROR_CUSPARSE;
	} else {
		return OPTKIT_SUCCESS;
	}
}

#ifdef __cplusplus
}
#endif


#endif /* OPTKIT_DEFS_GPU_H_ */
