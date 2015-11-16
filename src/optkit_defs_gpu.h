#ifndef OPTKIT_DEFS_GPU_H_GUARD
#define OPTKIT_DEFS_GPU_H_GUARD

#include "optkit_defs.h"

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


#define ok_free_gpu(x) \
  do { \
  	cudaFree(x); \
  	CUDA_CHECK_ERR; \
  	x = OK_NULL;
  } while(0)

#ifndef FLOAT
    #define CUBLAS(x) CUDA_CHECK_ERR; cublasD ## x
    #define CUSPARSE(x) CUDA_CHECK_ERR; cusparseD ## x
#else
    #define CUBLAS(x) CUDA_CHECK_ERR; cublasS ## x
    #define CUSPARSE(x) CUDA_CHECK_ERR; cusparseS ## x
#endif

inline unsigned int calc_grid_dim(size_t size) {
	return (unsigned int) fmin( ( (unsigned int) size + kBlockSize - 1u) 
									/ kBlockSize, kMaxGridSize);
}

#endif /* OPTKIT_DEFS_GPU_H_GUARD */