#include "optkit_prox.h"
#include "optkit_defs_gpu.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


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


/* CUDA helper kernels */
__global__ void __set_fn_vector(FunctionObj *objs, 
					const ok_float a, const ok_float b, const ok_float c,
					const ok_float d, const ok_float e, 
					const Function_t h, uint n) {
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = tid; i < (n; i += gridDim.x * blockDim.x)
		objs[i] = (FunctionObj){a, b, c, d, e, h};
}

/* CUDA C++ implementation with thrust:: */

/* thrust::binary function defining elementwise prox evaluation*/
struct ProxEvalF : thrust::binary_function<FunctionObj, ok_float, ok_float> {
  ok_float rho;
  __device__ ProxEvalF(ok_float rho) : rho(rho) { }
  __device__ ok_float operator()(const FunctionObj &f_obj, ok_float x) {
    return ProxEval(f_obj, x, rho);
  }
};

/* thrust::binary function defining elementwise function evaluation*/
struct FuncEvalF : thrust::binary_function<FunctionObj, ok_float, ok_float> {
  __device__ ok_float operator()(const FunctionObj &f_obj, T x) {
    return FuncEval(f_obj, x);
  }
};

/* vectorwise prox evaluation leveraging thrust::binary function */
void ProxEval_GPU(const FunctionVector * f, ok_float rho,
              const ok_float *x_in, size_t stride_in, 
              ok_float *x_out, size_t stride_out) {

	size_t N = f->size;
	strided_range<thrust::device_ptr<ok_float> > f_strided(
		thrust::device_pointer_cast(const_cast<FunctionObj *>(f->objectives)),
		thrust::device_pointer_cast(const_cast<FunctionObj *>(f->objectives) 
    								+ N, 1);

	strided_range<thrust::device_ptr<T> > x_in_strided(
		thrust::device_pointer_cast(const_cast<T*>(x_in)),
		thrust::device_pointer_cast(const_cast<T*>(x_in) + stride_in * N),
		stride_in);
	strided_range<thrust::device_ptr<T> > x_out_strided(
		thrust::device_pointer_cast(x_out),
		thrust::device_pointer_cast(x_out + stride_out * N), stride_out);

	thrust::transform(thrust::device, f_strided.begin(), f_strided.end(),
		x_in_strided.begin(), x_out_strided.begin(), ProxEvalF<T>(rho));
}

/* vectorwise function evaluation leveraging thrust::binary function */
ok_float FuncEval_GPU(const FunctionVector * f, const T *x_in, 
					  size_t stride) {

	size_t N = f->size;
	strided_range<thrust::device_ptr<ok_float> > f_strided(
		thrust::device_pointer_cast(const_cast<FunctionObj *>(f->objectives)),
		thrust::device_pointer_cast(const_cast<FunctionObj *>(f->objectives) 
    								+ N, 1);
	strided_range<thrust::device_ptr<ok_float> > x_in_strided(
		thrust::device_pointer_cast(const_cast<ok_float*>(x_in)),
		thrust::device_pointer_cast(const_cast<ok_float*>(x_in) + 
    								stride * N, stride);
	return thrust::inner_product(f_strided.begin(), f_strided.end(),
      							 x_in_strided.begin(), (ok_float) 0, 
      							 thrust::plus<ok_float>(), FuncEvalF());
}




/* CUDA C implementation to match header */
#ifdef __cplusplus
extern "C" {
#endif



__device__ inline void checkvexity(FunctionObj * f){
	if (f->c < (ok_float) 0){
		printf("WARNING: f not convex for c < 0
				(provided: %e). Using c = 0", f->c);	
		f->c = (ok_float) 0;	
	}
	if (f->e < (ok_float) 0){
		printf("WARNING: f not convex for e < 0
				(provided: %e). Using e = 0", f->e);	
		f->e = (ok_float) 0;	
	}
}



void function_vector_alloc(FunctionVector * f, size_t n){
	cudaError_t err;

	function_vector_free(f);
	f->size = len;
	err = cudaMalloc((void **) &f->objectives), n * sizeof(T));
	CUDA_CHECK_ERR;
	if (err != cudaSuccess) f->objectives = OK_NULL;
}

void function_vector_calloc(FunctionVector * f, size_t n){
	size_t i;
	uint grid_dim;

	function_vector_alloc(f, len)
	if (f->objectives != OK_NULL){
		grid_dim = calc_grid_dim(n);
		__set_fn_vector<<<grid_dim, kBlockSize>>>(f->objectives, 
							(ok_float) 1, (ok_float) 0, (ok_float) 1, 
							(ok_float) 0, (ok_float) 0, FnZero, n);
	}
}

void function_vector_free(FunctionVector * f){
	if (f->objectives != OK_NULL) ok_free_gpu(f->objectives);
}


void ProxEval(const * FunctionVector f, ok_float rho, 
			  const ok_float * x_in, size_t stride_in, 
			  ok_float * x_out, size_t stride_out) {

	ProxEval_GPU(f, rho, x_in, stride_in, x_out, stride_out);
}


ok_float FuncEval(const * FunctionVector f, const ok_float * x_in, 
				  size_t stride) {
	return FuncEval_GPU(f, x_in, stride);
}


#ifdef
}		/* extern "C" */
#endif