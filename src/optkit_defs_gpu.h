#ifndef OPTKIT_DEFS_GPU_H_GUARD
#define OPTKIT_DEFS_GPU_H_GUARD

#include "optkit_defs.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>

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

#define ok_alloc_gpu(x,n) \
  do { \
    cudaMalloc(x, n);
    CUDA_CHECK_ERR;
  } while (0)


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

inline uint 
calc_grid_dim(size_t size) {
	return (uint) fmin( ( (uint) size + kBlockSize - 1u) 
									/ kBlockSize, kMaxGridSize);
}

#ifdef __cplusplus
}
#endif

/* strided iterator from thrust:: examples. */
template <typename Iterable>
class strided_range {
 public:
  typedef typename thrust::iterator_difference<It>::type diff_t;

  struct StrideF : public thrust::unary_function<diff_t, diff_t> {
    diff_t stride;
    StrideF(diff_t stride) : stride(stride) { }
    __host__ __device__
    diff_t operator()(const diff_t& i) const { 
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<diff_t> CountingIt;
  typedef typename thrust::transform_iterator<StrideF, CountingIt> TransformIt;
  typedef typename thrust::permutation_iterator<Iterable, TransformIt> PermutationIt;
  typedef PermutationIt strided_iterator_t;

  /* construct strided_range for the range [first,last). */
  strided_range(Iterable first, Iterable last, diff_t stride)
      : first(first), last(last), stride(stride) { }
 
  strided_iterator_t begin() const {
    return PermutationIt(first, TransformIt(CountingIt(0), StrideF(stride)));
  }

  strided_iterator_t end() const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }
  
 protected:
  Iterable first;
  Iterable last;
  diff_t stride;
};

typedef thrust::constant_iterator<ok_float> constant_iterator_t;
typedef strided_range< thrust::device_ptr<ok_float> > strided_range_t;
typedef thrust::binary_function<ok_float, ok_float, ok_float> binary_function_t;
typedef thrust::binary_function<ok_float, ok_float, ok_float> binary_function_t;


#endif /* OPTKIT_DEFS_GPU_H_GUARD */