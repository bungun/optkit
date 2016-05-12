#include "optkit_prox.hpp"
#include "optkit_defs_gpu.h"
#include "optkit_thrust.hpp"

namespace optkit {

/* CUDA helper kernels */
template<typename T>
__global__ static void set_fn_vector(function_t_<T> * objs, const T a,
	const T b, const T c, const T d, const T e,
	const enum OPTKIT_SCALAR_FUNCTION h, uint n)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (uint i = tid; i < n; i += gridDim.x * blockDim.x) {
		objs[i].a = a;
		objs[i].b = b;
		objs[i].c = c;
		objs[i].d = d;
		objs[i].e = e;
		objs[i].h = h;
	}
}

/*
 * CUDA C++ implementation with thrust::
 * =====================================
 */

/* thrust::binary function defining elementwise prox evaluation*/
template<typename T>
struct ProxEvalF : thrust::binary_function<function_t_<T>, T, T>
{
	T rho;
	ProxEvalF(T rho) : rho(rho) { }
	__device__ T operator()(const function_t_<T> & f_obj, T x)
		{ return ProxEval<T>(&f_obj, x, rho); }
};

/* thrust::binary function defining elementwise function evaluation*/
template<typename T>
struct FuncEvalF : thrust::binary_function<function_t_<T>, T, T>
{
	__device__ T operator()(const function_t_<T> & f_obj, T x)
		{ return FuncEval<T>(&f_obj, x); }
};

} /* namespace optkit */

/* vectorwise prox evaluation leveraging thrust::binary function */
template<typename T>
void prox_eval_gpu(const function_t_<T> * f, T rho, const T * x_in,
	const size_t stride_in, T *x_out, const size_t stride_out, size_t n)
{
	strided_range<thrust::device_ptr<const function_t_<T> > > f_strided(
		thrust::device_pointer_cast(f),
		thrust::device_pointer_cast(f + n), 1);

	strided_range<thrust::device_ptr<const T> > x_in_strided(
		thrust::device_pointer_cast(x_in),
		thrust::device_pointer_cast(x_in + stride_in * n),
		stride_in);
	strided_range<thrust::device_ptr<T> > x_out_strided(
		thrust::device_pointer_cast(x_out),
		thrust::device_pointer_cast(x_out + stride_out * n),
		stride_out);
	thrust::transform(thrust::device, f_strided.begin(), f_strided.end(),
		x_in_strided.begin(), x_out_strided.begin(),
		optkit::ProxEvalF<T>(rho));
}

/* vectorwise function evaluation using thrust::binary_function */
template<typename T>
T function_eval_gpu(function_t_<T> * const f, T * const x, size_t stride,
	size_t n)
{
	strided_range<thrust::device_ptr<const function_t_<T> > > f_strided(
		thrust::device_pointer_cast(f),
		thrust::device_pointer_cast(f + n), 1);
	strided_range<thrust::device_ptr<const T> > x_strided(
		thrust::device_pointer_cast(x),
		thrust::device_pointer_cast(x + stride * n), stride);
	return thrust::inner_product(f_strided.begin(), f_strided.end(),
		x_strided.begin(), static_cast<T>(0), thrust::plus<T>(),
		optkit::FuncEvalF<T>());
}

template<typename T>
ok_status function_vector_alloc_(function_vector_<T> * f, size_t n)
{
	OK_CHECK_PTR(f);
	if (f->objectives)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	f->size = n;
	return ok_alloc_gpu(f->objectives, n * sizeof(*f->objectives));
}

template<typename T>
ok_status function_vector_calloc_(function_vector_<T> * f, size_t n)
{
	uint grid_dim;

	OK_RETURNIF_ERR( function_vector_alloc(f, n) );

	grid_dim = calc_grid_dim(n);
	optkit::set_fn_vector<<<grid_dim, kBlockSize>>>(f->objectives,
		static_cast<T>(1), static_cast<T>(0), static_cast<T>(1),
		static_cast<T>(0), static_cast<T>(0), FnZero, n);
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

template<typename T>
ok_status function_vector_free_(function_vector_<T> * f)
{
	OK_CHECK_FNVECTOR(f);
	f->size = 0;
	return ok_free_gpu(f->objectives);
}

template<typename T>
ok_status function_vector_view_array_(function_vector_<T> * f,
	function_t_<T> * h, size_t n)
{
	OK_CHECK_PTR(f);
	OK_CHECK_PTR(h);
	f->size = n;
	f->objectives = h;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status function_vector_memcpy_va_(function_vector_<T> * f,
	function_t_<T> * h)
{
	return ok_memcpy_gpu(f->objectives, h, f->size * sizeof(function_t));
}

template<typename T>
ok_status function_vector_memcpy_av_(function_t_<T> * h,
	function_vector_<T> * f)
{
	return ok_memcpy_gpu(h, f->objectives, f->size * sizeof(function_t));
}

template<typename T>
ok_status function_vector_mul_(function_vector_<T> * f, const vector_<T> * v)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(v);

	vector el;

	if (f->size != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	el.size = f->size;
	el.stride = sizeof(function_t) / sizeof(T);

	el.data = &(f->objectives->a);
	OK_RETURNIF_ERR( vector_mul(&el, v) );

	el.data = &(f->objectives->d);
	OK_RETURNIF_ERR( vector_mul(&el, v) );

	el.data = &(f->objectives->e);
	return OK_SCAN_ERR( vector_mul(&el, v) );
}

template<typename T>
ok_status function_vector_div_(function_vector_<T> * f, const vector_<T> * v)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(v);
	vector el;

	el.size = f->size;
	el.stride = sizeof(function_t) / sizeof(T);

	el.data = &(f->objectives->a);
	OK_RETURNIF_ERR( vector_div(&el, v) );

	el.data = &(f->objectives->d);
	OK_RETURNIF_ERR( vector_div(&el, v) );

	el.data = &(f->objectives->e);
	return OK_SCAN_ERR( vector_div(&el, v) );
}

template<typename T>
ok_status function_vector_print_(function_vector_<T> * f)
{
	size_t i;
	ok_status err = OPTKIT_SUCCESS;
	const char * fmt =
		"h: %i, a: %0.2e, b: %0.2e, c: %0.2e, d: %0.2e, e: %0.2e\n";
	OK_CHECK_FNVECTOR(f);

	function_t obj_host[f->size];
	err = ok_memcpy_gpu(&obj_host, f->objectives, f->size * sizeof(function_t));
	if (!err)
		for (i = 0; i < f->size; ++i)
			printf(fmt, (int) obj_host[i].h, obj_host[i].a,
				obj_host[i].b, obj_host[i].c, obj_host[i].d,
				obj_host[i].e);
	return err;
}

template<typename T>
ok_status prox_eval_vector_(const function_vector_<T> * f, T rho,
	const vector_<T> * x_in, vector_<T> * x_out)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(x_in);
	OK_CHECK_VECTOR(x_out);
	if (f->size != x_in->size || f->size != x_out->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	if (rho <= 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_DOMAIN );
	prox_eval_gpu<T>(f->objectives, rho, x_in->data, x_in->stride,
		x_out->data, x_out->stride, f->size);
	return OK_STATUS_CUDA;
}

template<typename T>
ok_status function_eval_vector_(const function_vector_<T> * f,
	const vector_<T> * x, T * fn_val)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_VECTOR(x);
	OK_CHECK_PTR(fn_val);
	if (f->size != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	*fn_val = function_eval_gpu<T>(f->objectives, x->data, x->stride,
		f->size);
	return OK_STATUS_CUDA;
}

#ifdef __cplusplus
extern "C" {
#endif

ok_status function_vector_alloc(function_vector * f, size_t n)
	{ return function_vector_alloc_<ok_float>(f, n); }

ok_status function_vector_calloc(function_vector * f, size_t n)
	{ return function_vector_calloc_<ok_float>(f, n); }

ok_status function_vector_free(function_vector * f)
	{ return function_vector_free_<ok_float>(f); }

ok_status function_vector_view_array(function_vector * f, function_t * h,
	size_t n)
	{ return function_vector_view_array_<ok_float>(f, h, n); }

ok_status function_vector_memcpy_va(function_vector * f, function_t * h)
	{ return function_vector_memcpy_va_<ok_float>(f, h); }

ok_status function_vector_memcpy_av(function_t * h, function_vector * f)
	{ return function_vector_memcpy_av_<ok_float>(h, f); }

ok_status function_vector_mul(function_vector * f, const vector * v)
	{ return function_vector_mul_<ok_float>(f, v); }

ok_status function_vector_div(function_vector * f, const vector * v)
	{ return function_vector_div_<ok_float>(f, v); }

ok_status function_vector_print(function_vector *f)
	{ return function_vector_print_<ok_float>(f); }

ok_status prox_eval_vector(const function_vector * f, ok_float rho,
	const vector * x_in, vector * x_out)
	{ return prox_eval_vector_<ok_float>(f, rho, x_in, x_out); }
ok_status function_eval_vector(const function_vector * f, const vector * x,
	ok_float * fn_val)
	{ return function_eval_vector_<ok_float>(f, x, fn_val); }

#ifdef __cplusplus
}
#endif
