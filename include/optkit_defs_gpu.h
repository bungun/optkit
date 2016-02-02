#ifndef OPTKIT_DEFS_GPU_H_GUARD
#define OPTKIT_DEFS_GPU_H_GUARD

#include "optkit_defs.h"
#include <math_constants.h>

#ifdef __cplusplus
extern "C" {
#endif

const unsigned int kTileSize = 32u;
const unsigned int kBlockSize = 256u;
const unsigned int kMaxGridSize = 65535u;

#ifndef CUDA_CHECK_ERR
#define CUDA_CHECK_ERR \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      printf("%s:%d:%s\n ERROR_CUDA: %s\n", __FILE__, __LINE__, __func__, \
             cudaGetErrorString(err)); \
    } \
  } while (0)
#endif

#define ok_alloc_gpu(x, n) \
  do { \
    cudaMalloc((void **) &x, n); \
    CUDA_CHECK_ERR; \
  } while (0)

#define ok_memcpy_gpu(x, y, n) \
  do { \
    cudaMemcpy(x, y, n, cudaMemcpyDefault); \
    CUDA_CHECK_ERR; \
  } while (0)

#define ok_free_gpu(x) \
  do { \
  	cudaFree(x); \
  	CUDA_CHECK_ERR; \
  	x = OK_NULL; \
  } while(0)

#ifndef FLOAT
    #define CUBLAS(x) CUDA_CHECK_ERR; cublasD ## x
    #define CUSPARSE(x) CUDA_CHECK_ERR; cusparseD ## x
    #define OK_CUDA_NAN CUDART_NAN
#else
    #define CUBLAS(x) CUDA_CHECK_ERR; cublasS ## x
    #define CUSPARSE(x) CUDA_CHECK_ERR; cusparseS ## x
    #define OK_CUDA_NAN CUDART_NAN_F
#endif


inline uint 
calc_grid_dim(size_t size) {
	return (uint) min( ( (uint) size + kBlockSize - 1u) 
									/ kBlockSize, kMaxGridSize);
}

#ifdef __cplusplus
}
#endif



#endif /* OPTKIT_DEFS_GPU_H_GUARD */