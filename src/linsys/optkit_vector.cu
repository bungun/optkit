#include "optkit_defs_gpu.h"
#include "optkit_thrust.hpp"
#include "optkit_vector.h"

/* CUDA helper methods */
template<typename T>
static __global__ void __vector_set(T * data, T val, size_t stride,
	size_t size)
{
	uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
	data[i * stride] = val;
}

template<typename T>
static void __vector_set_all(vector_<T> * v, T x)
{
	uint grid_dim = calc_grid_dim(v->size);
	__vector_set<T><<<grid_dim, kBlockSize>>>(v->data, x, v->stride, v->size);
}

template<typename T>
static __global__ void __strided_memcpy(T * x, size_t stride_x, const T * y,
	size_t stride_y, size_t size)
{
	uint i, tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = tid; i < size; i += gridDim.x * blockDim.x)
	x[i * stride_x] = y[i * stride_y];
}

template<typename T>
void vector_alloc_(vector_<T> * v, size_t n)
{
	if (!v || !v->data)
		return;
	v->size = n;
	v->stride = 1;
	ok_alloc_gpu(v->data, n * sizeof(T));
}

template<typename T>
void vector_calloc_(vector_<T> * v, size_t n)
{
	vector_alloc_<T>(v, n);
	__vector_set_all<T>(v, static_cast<T>(0));
}

template<typename T>
void vector_free_(vector_<T> * v)
{
	if (v && v->data != OK_NULL)
		ok_free_gpu(v->data);
	v->size = (size_t) 0;
	v->stride = (size_t) 0;
}

template<typename T>
void vector_set_all_(vector_<T> * v, T x)
{
	__vector_set_all(v, x);
}

template<typename T>
void vector_subvector_(vector_<T> * v_out, vector_<T> * v_in, size_t offset,
	size_t n)
{
	if (!v_out || !v_in)
		return;
	v_out->size=n;
	v_out->stride=v_in->stride;
	v_out->data=v_in->data + offset * v_in->stride;
}

template<typename T>
void vector_view_array_(vector_<T> * v, T * base, size_t n)
{
	  if (!v)
		return;
	  v->size=n;
	  v->stride=1;
	  v->data=base;
}

template<typename T>
void vector_memcpy_vv_(vector_<T> * v1, const vector_<T> * v2)
{
	uint grid_dim;
	if ( v1->stride == 1 && v2->stride == 1) {
		ok_memcpy_gpu(v1->data, v2->data, v1->size * sizeof(T));
	} else {
		grid_dim = calc_grid_dim(v1->size);
		__strided_memcpy<T><<<grid_dim, kBlockSize>>>(v1->data,
			v1->stride, v2->data, v2->stride, v1->size);
	}
}

template<typename T>
void vector_memcpy_va_(vector_<T> * v, const T *y, size_t stride_y)
{
	uint i;
	if (v->stride == 1 && stride_y == 1)
		ok_memcpy_gpu(v->data, y, v->size * sizeof(T));
	else
		for (i = 0; i < v->size; ++i)
			ok_memcpy_gpu(v->data + i * v->stride, y + i * stride_y,
				sizeof(T));
}

template<typename T>
void vector_memcpy_av_(T *x, const vector_<T> *v, size_t stride_x)
{
	uint i;
	if (v->stride == 1 && stride_x == 1)
		ok_memcpy_gpu(x, v->data, v->size * sizeof(T));
	else
		for (i = 0; i < v->size; ++i)
			ok_memcpy_gpu(x + i * stride_x, v->data + i * v->stride,
				sizeof(T));
}

#ifdef __cplusplus
extern "C" {
#endif

void vector_alloc(vector * v, size_t n)
	{ vector_alloc_<ok_float>(v, n); }

void vector_calloc(vector * v, size_t n)
	{ vector_calloc_<ok_float>(v, n); }

void vector_free(vector * v)
	{ vector_free_<ok_float>(v); }

void vector_set_all(vector * v, ok_float x)
	{ vector_set_all_<ok_float>(v, x); }

void vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n)
	{ vector_subvector_<ok_float>(v_out, v_in, offset, n); }

void vector_view_array(vector * v, ok_float * base, size_t n)
	{ vector_view_array_<ok_float>(v, base, n); }

void vector_memcpy_vv(vector * v1, const vector * v2)
	{ vector_memcpy_vv_<ok_float>(v1, v2); }

void vector_memcpy_va(vector * v, const ok_float *y, size_t stride_y)
	{ vector_memcpy_va_<ok_float>(v, y, stride_y); }

void vector_memcpy_av(ok_float * x, const vector * v, size_t stride_x)
	{ vector_memcpy_av_<ok_float>(x, v, stride_x); }

void vector_print(const vector * v)
{
	uint i;
	ok_float v_host[v->size];
	vector_memcpy_av(v_host, v, 1);
	for (i = 0; i < v->size; ++i)
		printf("%e ", v_host[i]);
	printf("\n");
}

void vector_scale(vector * v, ok_float x)
{
	__thrust_vector_scale(v, x);
	CUDA_CHECK_ERR;
}

void vector_add(vector * v1, const vector * v2)
{
	__thrust_vector_add(v1, v2);
	CUDA_CHECK_ERR;
}

void vector_sub(vector * v1, const vector * v2)
{
	__thrust_vector_sub(v1, v2);
	CUDA_CHECK_ERR;
}

void vector_mul(vector * v1, const vector * v2)
{
	__thrust_vector_mul(v1, v2);
	CUDA_CHECK_ERR;
}

void vector_div(vector * v1, const vector * v2)
{
	__thrust_vector_div(v1, v2);
	CUDA_CHECK_ERR;
}

void vector_add_constant(vector * v, const ok_float x)
{
	__thrust_vector_add_constant(v, x);
	CUDA_CHECK_ERR;
}

void vector_abs(vector * v)
{
	__thrust_vector_abs(v);
	CUDA_CHECK_ERR;
}

void vector_recip(vector * v)
{
	__thrust_vector_recip(v);
	CUDA_CHECK_ERR;
}

void vector_safe_recip(vector * v)
{
	__thrust_vector_safe_recip(v);
	CUDA_CHECK_ERR;
}

void vector_sqrt(vector * v)
{
	__thrust_vector_sqrt(v);
	CUDA_CHECK_ERR;
}

void vector_pow(vector * v, const ok_float x)
{
	__thrust_vector_pow(v, x);
	CUDA_CHECK_ERR;
}

void vector_exp(vector * v)
{
	__thrust_vector_exp(v);
	CUDA_CHECK_ERR;
}

size_t vector_indmin(vector * v)
{
	size_t minind = __thrust_vector_indmin<ok_float>(v);
	CUDA_CHECK_ERR;
	return minind;
}

ok_float vector_min(vector * v)
{
	ok_float minval = __thrust_vector_min<ok_float>(v);
	CUDA_CHECK_ERR;
	return minval;
}

ok_float vector_max(vector * v)
{
	ok_float maxval = __thrust_vector_max<ok_float>(v);
	CUDA_CHECK_ERR;
	return maxval;
}

void indvector_alloc(indvector * v, size_t n)
	{ vector_alloc_<size_t>(v, n); }

void indvector_calloc(indvector * v, size_t n)
	{ vector_calloc_<size_t>(v, n); }

void indvector_free(indvector * v)
	{ vector_free_<size_t>(v); }

void indvector_set_all(indvector * v, size_t x)
	{ vector_set_all_<size_t>(v, x); }

void indvector_subvector(indvector * v_out, indvector * v_in, size_t offset,
	size_t n)
	{ vector_subvector_<size_t>(v_out, v_in, offset, n); }

void indvector_view_array(indvector * v, size_t * base, size_t n)
	{ vector_view_array_<size_t>(v, base, n); }

void indvector_memcpy_vv(indvector * v1, const indvector * v2)
	{ vector_memcpy_vv_<size_t>(v1, v2); }

void indvector_memcpy_va(indvector * v, const size_t * y, size_t stride_y)
	{ vector_memcpy_va_<size_t>(v, y, stride_y); }

void indvector_memcpy_av(size_t * x, const indvector * v, size_t stride_x)
	{ vector_memcpy_av_<size_t>(x, v, stride_x); }

#ifdef __cplusplus
}
#endif