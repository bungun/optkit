#include "optkit_defs_gpu.h"
#include "optkit_thrust.hpp"
#include "optkit_vector.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA helper methods */
__global__ void __vector_set(ok_float * data, ok_float val, size_t stride,
	size_t size)
{
	uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
	data[i * stride] = val;
}

void __vector_set_all(vector * v, ok_float x)
{
	uint grid_dim = calc_grid_dim(v->size);
	__vector_set<<<grid_dim, kBlockSize>>>(v->data, x, v->stride, v->size);
}

__global__ void __strided_memcpy(ok_float * x, size_t stride_x,
	const ok_float * y, size_t stride_y, size_t size)
{
	uint i, tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = tid; i < size; i += gridDim.x * blockDim.x)
	x[i * stride_x] = y[i * stride_y];
}

/* vector methods */
inline int __vector_exists(vector * v)
{
	if (v == OK_NULL) {
		printf("Error: cannot write to uninitialized vector pointer\n");
		return 0;
	} else {
		return 1;
	}
}

void vector_alloc(vector * v, size_t n)
{
	if (!__vector_exists(v))
		return;
	v->size=n;
	v->stride=1;
	ok_alloc_gpu(v->data, n * sizeof(ok_float));
}

void vector_calloc(vector * v, size_t n)
{
	vector_alloc(v, n);
	__vector_set_all(v, ok_float(0));
}

void vector_free(vector * v)
{
	if (v != OK_NULL)
		if (v->data != OK_NULL) ok_free_gpu(v->data);
	v->size = (size_t) 0;
	v->stride = (size_t) 0;
}

void vector_set_all(vector * v, ok_float x)
{
	__vector_set_all(v, x);
}

void vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n)
{
	if (!__vector_exists(v_out))
		return;
	v_out->size=n;
	v_out->stride=v_in->stride;
	v_out->data=v_in->data + offset * v_in->stride;
}

vector vector_subvector_gen(vector * v_in, size_t offset, size_t n)
{
	return (vector){
		.size = n,
		.stride = v_in->stride,
		.data = v_in->data + offset * v_in->stride
	};
}

void vector_view_array(vector * v, ok_float * base, size_t n)
{
	  if (!__vector_exists(v))
		return;
	  v->size=n;
	  v->stride=1;
	  v->data=base;
}


void vector_memcpy_vv(vector * v1, const vector * v2)
{
	uint grid_dim;
	if ( v1->stride == 1 && v2->stride == 1) {
		ok_memcpy_gpu(v1->data, v2->data, v1->size * sizeof(ok_float));
	} else {
		grid_dim = calc_grid_dim(v1->size);
		__strided_memcpy<<<grid_dim, kBlockSize>>>(v1->data, v1->stride,
			v2->data, v2->stride, v1->size);
	}
}

void vector_memcpy_va(vector * v, const ok_float *y, size_t stride_y)
{
	uint i;
	if (v->stride == 1 && stride_y == 1)
		ok_memcpy_gpu(v->data, y, v->size * sizeof(ok_float));
	else
		for (i = 0; i < v->size; ++i)
	ok_memcpy_gpu(v->data + i * v->stride, y + i * stride_y,
	sizeof(ok_float));
}

void vector_memcpy_av(ok_float *x, const vector *v, size_t stride_x)
{
	uint i;
	if (v->stride == 1 && stride_x == 1)
		ok_memcpy_gpu(x, v->data, v->size * sizeof(ok_float));
	else
		for (i = 0; i < v->size; ++i)
	ok_memcpy_gpu(x + i * stride_x, v->data + i * v->stride,
	sizeof(ok_float));
}

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
	__thrust_vector_exp(v, x);
	CUDA_CHECK_ERR;
}

size_t vector_indmin(vector * v)
{
	size_t minind = __thrust_vector_indmin<ok_float>(v, x);
	CUDA_CHECK_ERR;
	return minind;
}

ok_float vector_min(vector * v)
{
	ok_float minval = __thrust_vector_min<ok_float>(v, x);
	CUDA_CHECK_ERR;
	return minval;
}

ok_float vector_max(vector * v)
{
	ok_float maxval = __thrust_vector_max<ok_float>(v, x);
	CUDA_CHECK_ERR;
	return maxval;
}

#ifdef __cplusplus
}
#endif