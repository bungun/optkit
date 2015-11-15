#ifndef OPTKIT_DEFS_H_GUARD
#define OPTKIT_DEFS_H_GUARD

#include <tgmath.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gsl_cblas.h"

#ifdef __CUDACC__
#include <cublas_v2.h>
/* #include <cusparse.h> */
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define OK_NULL 0
#define ok_free(x) free(x); x=OK_NULL; printf("variable freed\n")



typedef enum CBLAS_ORDER CBLAS_ORDER_t;
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE_t;
typedef enum CBLAS_UPLO CBLAS_UPLO_t;
typedef enum CBLAS_DIAG CBLAS_DIAG_t;
typedef enum CBLAS_SIDE CBLAS_SIDE_t;

#ifndef FLOAT
    #define CBLAS(x) cblas_d ## x
    typedef double ok_float;
    #define MACHINETOL (ok_float) 10e-10
#else
    #define CBLAS(x) cblas_s ## x
    typedef float ok_float;
    #define MACHINETOL (ok_float) 10e-5
#endif


#ifdef __CUDACC__

const unsigned int kTileSize = 32u;
const unsigned int kBlockSize = 256u;
const unsigned int kMaxGridSize = 65535u;

#define CUDA_CHECK_ERR \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      printf("%s:%d:%s\n ERROR_CUDA: %s\n", __FILE__, __LINE__, __func__, \
             cudaGetErrorString(err)); \
    } \
  } while (0)


#ifndef FLOAT
    #define CUBLAS(x) CUDA_CHECK_ERR; cublasD ## x
    #define CUSPARSE(x) CUDA_CHECK_ERR; cusparseD ## x
#else
    #define CUBLAS(x) CUDA_CHECK_ERR; cublasS ## x
    #define CUSPARSE(x) CUDA_CHECK_ERR; cusparseS ## x
#endif

inline unsigned int calc_grid_dim(size_t size, unsigned int block_size) {
	return (unsigned int) fmin( ( (unsigned int) size + block_size - 1u) 
									/ block_size, kMaxGridSize);
}



#endif /* __CUDACC__ */


#ifdef __cplusplus
}
#endif

#endif /* GSL_DEFS_H_GUARD */
