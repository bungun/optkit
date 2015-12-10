#include "optkit_prox.h"
#include "optkit_defs_gpu.h"



/* CUDA helper kernels */
__global__ void
__set_fn_vector_from_multi(FunctionObj * objs, const ok_float * a, 
	const ok_float * b, const ok_float * c, const ok_float * d,
	const ok_float * e, const Function_t * h, uint n){
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = tid; i < n; i += gridDim.x * blockDim.x){
		if (a != OK_NULL) 
			objs[i].a = a[i]; 
		if (b != OK_NULL) 
			objs[i].b = b[i]; 
		if (c != OK_NULL) 
			objs[i].c = c[i]; 
		if (d != OK_NULL) 
			objs[i].d = d[i]; 
		if (e != OK_NULL) 
		objs[i].e = e[i];
		if (h != OK_NULL) 
			objs[i].h = h[i];
	};
}


__global__ void 
__set_fn_vector(FunctionObj * objs, 
					const ok_float a, const ok_float b, const ok_float c,
					const ok_float d, const ok_float e, 
					const Function_t h, uint n) {
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = tid; i < n; i += gridDim.x * blockDim.x)
		objs[i] = (FunctionObj){
			.a = a, 
			.b = b, 
			.c = c, 
			.d = d, 
			.e = e, 
			.h = h
		};
}

/* CUDA C++ implementation with thrust:: */

/* thrust::binary function defining elementwise prox evaluation*/
struct ProxEvalF : thrust::binary_function<FunctionObj, ok_float, ok_float> {
  ok_float rho;
  ProxEvalF(ok_float rho) : rho(rho) { }
  __device__ ok_float operator()(const FunctionObj & f_obj, ok_float x) {
    return ProxEval(&f_obj, x, rho);
  }
};


/* thrust::binary function defining elementwise function evaluation*/
struct FuncEvalF : thrust::binary_function<FunctionObj, ok_float, ok_float> {
  __device__ ok_float operator()(const FunctionObj & f_obj, ok_float x) {
    return FuncEval(&f_obj, x);
  }
};


/* vectorwise prox evaluation leveraging thrust::binary function */
void 
ProxEval_GPU(const FunctionObj * f, ok_float rho, 
	const ok_float * x_in, const size_t stride_in, 
	ok_float *x_out, const size_t stride_out, size_t n){
	strided_range<thrust::device_ptr<const FunctionObj> > f_strided(
		thrust::device_pointer_cast(f),
		thrust::device_pointer_cast(f + n), 1);

	strided_range<thrust::device_ptr<const ok_float> > x_in_strided(
		thrust::device_pointer_cast(x_in),
		thrust::device_pointer_cast(x_in + stride_in * n), stride_in);
	strided_range<thrust::device_ptr<ok_float> > x_out_strided(
		thrust::device_pointer_cast(x_out),
		thrust::device_pointer_cast(x_out + stride_out * n), stride_out);

	thrust::transform(thrust::device, f_strided.begin(), f_strided.end(),
		x_in_strided.begin(), x_out_strided.begin(), ProxEvalF(rho));
}

/* vectorwise function evaluation using thrust::binary_function */
void 
FuncEval_GPU(const FunctionObj * f, const ok_float * x, 
	size_t stride, size_t n, ok_float * res) {
	strided_range<thrust::device_ptr<const FunctionObj> > f_strided(
		thrust::device_pointer_cast(f),
		thrust::device_pointer_cast(f + n), 1);
	strided_range<thrust::device_ptr<const ok_float> > x_strided(
		thrust::device_pointer_cast(x),
		thrust::device_pointer_cast(x + stride * n), stride);
	* res = thrust::inner_product(f_strided.begin(), f_strided.end(),
      							 x_strided.begin(), (ok_float) 0, 
      							 thrust::plus<ok_float>(), FuncEvalF());
}


/* CUDA C implementation to match header */
#ifdef __cplusplus
extern "C" {
#endif



__device__ inline void 
checkvexity(FunctionObj * f){
	if (f->c < (ok_float) 0){
		printf("WARNING: f not convex for c < 0 (provided: %e). Using c = 0" \
			, f->c);	
		f->c = (ok_float) 0;	
	}
	if (f->e < (ok_float) 0){
		printf("WARNING: f not convex for e < 0 (provided: %e). Using e = 0" \
			, f->e);	
		f->e = (ok_float) 0;	
	}
}


void 
function_vector_alloc(FunctionVector * f, size_t n){
	function_vector_free(f);
	f->size = n;
	ok_alloc_gpu( f->objectives, n * sizeof( FunctionObj ) );
	CUDA_CHECK_ERR;
	//if (err != cudaSuccess) f->objectives = OK_NULL;
}

void 
function_vector_calloc(FunctionVector * f, size_t n){
	uint grid_dim;

	function_vector_alloc(f, n);
	if (f->objectives != OK_NULL) {
		grid_dim = calc_grid_dim(n);
		__set_fn_vector<<<grid_dim, kBlockSize>>>(f->objectives, 
							(ok_float) 1, (ok_float) 0, (ok_float) 1, 
							(ok_float) 0, (ok_float) 0, FnZero, n);
	}
}

void 
function_vector_free(FunctionVector * f){
	if (f->objectives != OK_NULL) ok_free_gpu(f->objectives);
}


void function_vector_view_array(FunctionVector * f, 
                                FunctionObj * h, size_t n){
	f->size=n;
	f->objectives= (FunctionObj *) h;
}

void 
function_vector_from_multiarray(FunctionVector * f, Function_t * h, 
									 ok_float * a, ok_float * b, 
									 ok_float * c, ok_float * d, 
									 ok_float * e, size_t n){
	f->size=n;
	function_vector_alloc(f, n);
	function_vector_memcpy_vmulti(f, h, a, b, c, d, e);
}


void 
function_vector_memcpy_va(FunctionVector * f, FunctionObj * h){
	ok_memcpy_gpu(f->objectives, h, f->size * sizeof(FunctionObj));
}



void 
function_vector_memcpy_vmulti(FunctionVector * f, Function_t *h,
									 ok_float * a, ok_float * b, 
									 ok_float * c, ok_float * d, 
									 ok_float * e){


	uint grid_dim = calc_grid_dim(f->size);
	ok_float * a_dev, * b_dev, *c_dev, *d_dev, *e_dev;
	Function_t * h_dev = OK_NULL;

	a_dev = b_dev = c_dev = d_dev = e_dev = OK_NULL;

	if (h != OK_NULL)
		ok_memcpy_gpu(h_dev, h, f->size * sizeof(Function_t));
	if (a != OK_NULL)
		ok_memcpy_gpu(a_dev, a, f->size * sizeof(ok_float));
	if (b != OK_NULL)
		ok_memcpy_gpu(b_dev, b, f->size * sizeof(ok_float));
	if (c != OK_NULL)
		ok_memcpy_gpu(c_dev, c, f->size * sizeof(ok_float));
	if (d != OK_NULL)
		ok_memcpy_gpu(d_dev, d, f->size * sizeof(ok_float));
	if (e != OK_NULL)
		ok_memcpy_gpu(e_dev, e, f->size * sizeof(ok_float));

	__set_fn_vector_from_multi<<<grid_dim, kBlockSize>>>(f->objectives, 
		a_dev, b_dev, c_dev, d_dev, e_dev, h_dev, f->size);

}


void function_vector_print(FunctionVector * f){
	size_t i;
	FunctionObj * obj_host = (FunctionObj *) malloc(f->size * sizeof(FunctionObj));
	ok_memcpy_gpu(obj_host, f->objectives, f->size);
	for (i = 0; i < f->size; ++i)
		printf("h: %i, a: %0.2e, b: %0.2e, c: %0.2e, d: %0.2e, e: %0.2e\n", 
				(int) obj_host[i].h, obj_host[i].a, obj_host[i].b, 
				obj_host[i].c, obj_host[i].d, obj_host[i].e);
	ok_free(obj_host);
}


void 
ProxEvalVector(const FunctionVector * f, ok_float rho,
			  const vector * x_in, vector * x_out){
	ProxEval_GPU(f->objectives, rho, x_in->data, x_in->stride,
		x_out->data, x_out->stride, f->size);
}

ok_float 
FuncEvalVector(const FunctionVector * f, const vector * x){
	ok_float res;
	ok_float * res_dev;
	ok_alloc_gpu(res_dev, sizeof(ok_float));
	FuncEval_GPU(f->objectives, x->data, x->stride, 
		f->size, res_dev);
	ok_memcpy_gpu(&res, res_dev, sizeof(ok_float));
	return res;
}


#ifdef __cplusplus
}		/* extern "C" */
#endif