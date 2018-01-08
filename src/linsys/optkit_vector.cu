#include <curand_kernel.h>
#include "optkit_defs_gpu.h"
#include "optkit_thrust.hpp"
#include "optkit_vector.h"

/* CUDA helper methods */

namespace optkit {

static __global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}

static __global__ void generate(curandState *globalState, ok_float *data,
	const size_t size, const size_t stride)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x, i;
	#ifndef FLOAT
	for (i = tid; i < size; i += gridDim.x * blockDim.x)
		data[i * stride] = curand_uniform_double(globalState + tid);
	#else
	for (i = tid; i < size; i += gridDim.x * blockDim.x)
		data[i * stride] = curand_uniform(globalState + tid);
	#endif
}

} /* namespace optkit */

static ok_status ok_rand_u01(ok_float *x, const size_t size,
	const size_t stride)
{
	const size_t num_rand = size <= kMaxGridSize ? size : kMaxGridSize;
	curandState *device_states;
	int grid_dim;

	OK_CHECK_PTR(x);
	OK_RETURNIF_ERR( ok_alloc_gpu(device_states, num_rand *
		sizeof(*device_states)) );

	grid_dim = calc_grid_dim(num_rand);

	optkit::setup_kernel<<<grid_dim, kBlockSize>>>(device_states, 0);
	OK_RETURNIF_ERR( OK_STATUS_CUDA );

	optkit::generate<<<grid_dim, kBlockSize>>>(device_states, x, size,
		stride);
	OK_RETURNIF_ERR( OK_STATUS_CUDA );

	return OK_SCAN_ERR( ok_free_gpu(device_states) );
}


template<typename T>
static __global__ void __vector_set(T *data, T val, size_t stride, size_t size)
{
	uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
		data[i * stride] = val;
}

template<typename T>
static ok_status __vector_set_all(vector_<T> *v, T x)
{
	uint grid_dim = calc_grid_dim(v->size);
	__vector_set<T><<<grid_dim, kBlockSize>>>(v->data, x, v->stride, v->size);
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

template<typename T>
static __global__ void __strided_memcpy(T *x, size_t stride_x, const T *y,
	size_t stride_y, size_t size)
{
	uint i, tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = tid; i < size; i += gridDim.x * blockDim.x)
	x[i * stride_x] = y[i * stride_y];
}

template<typename T>
ok_status vector_alloc_(vector_<T> *v, size_t n)
{
	OK_CHECK_PTR(v);
	if (v->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	v->size = n;
	v->stride = 1;
	return ok_alloc_gpu(v->data, n * sizeof(T));
}

template<typename T>
ok_status vector_calloc_(vector_<T> *v, size_t n)
{
	OK_RETURNIF_ERR( vector_alloc_<T>(v, n) );
	return __vector_set_all<T>(v, static_cast<T>(0));
}

template<typename T>
ok_status vector_free_(vector_<T> *v)
{
	OK_CHECK_VECTOR(v);
	v->size = (size_t) 0;
	v->stride = (size_t) 0;
	return ok_free_gpu(v->data);
}

template<typename T>
ok_status vector_set_all_(vector_<T> *v, T x)
{
	return __vector_set_all(v, x);
}

template<typename T>
ok_status vector_memcpy_vv_(vector_<T> *v1, const vector_<T> *v2)
{
	uint grid_dim;
	OK_CHECK_VECTOR(v1);
	OK_CHECK_VECTOR(v2);
	if (v1->size != v2->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if ( v1->stride == 1 && v2->stride == 1) {
		return ok_memcpy_gpu(v1->data, v2->data, v1->size * sizeof(T));
	} else {
		grid_dim = calc_grid_dim(v1->size);
		__strided_memcpy<T><<<grid_dim, kBlockSize>>>(v1->data,
			v1->stride, v2->data, v2->stride, v1->size);
		cudaDeviceSynchronize();
		return OK_STATUS_CUDA;
	}
}

template<typename T>
ok_status vector_memcpy_va_(vector_<T> *v, const T *y, size_t stride_y)
{
	ok_status err = OPTKIT_SUCCESS;
	uint i;
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(y);

	if (v->stride == 1 && stride_y == 1)
		return ok_memcpy_gpu(v->data, y, v->size * sizeof(T));
	else
		for (i = 0; i < v->size && !err; ++i)
			err = ok_memcpy_gpu(v->data + i * v->stride,
				y + i * stride_y, sizeof(T));
	return err;
}

template<typename T>
ok_status vector_memcpy_av_(T *x, const vector_<T> *v, size_t stride_x)
{
	ok_status err = OPTKIT_SUCCESS;
	uint i;
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(x);

	if (v->stride == 1 && stride_x == 1)
		return ok_memcpy_gpu(x, v->data, v->size * sizeof(T));
	else
		for (i = 0; i < v->size && !err; ++i)
			err = ok_memcpy_gpu(x + i * stride_x,
				v->data + i * v->stride, sizeof(T));
	return err;
}

template<typename T>
ok_status vector_scale_(vector_<T> *v, T x)
{
        OK_CHECK_VECTOR(v);
        __thrust_vector_scale<T>(v, x);
        return OK_STATUS_CUDA;
}

template<typename T>
ok_status vector_add_(vector_<T> *v1, const vector_<T> *v2)
{
        OK_CHECK_VECTOR(v1);
        OK_CHECK_VECTOR(v2);
        if (v1->size != v2->size)
                return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

        __thrust_vector_add<T>(v1, v2);
        return OK_STATUS_CUDA;
}

template<typename T>
ok_status vector_sub_(vector_<T> *v1, const vector_<T> *v2)
{
        OK_CHECK_VECTOR(v1);
        OK_CHECK_VECTOR(v2);
        if (v1->size != v2->size)
                return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

        __thrust_vector_sub<T>(v1, v2);
        return OK_STATUS_CUDA;
}

template<typename T>
ok_status vector_mul_(vector_<T> *v1, const vector_<T> *v2)
{
        OK_CHECK_VECTOR(v1);
        OK_CHECK_VECTOR(v2);
        if (v1->size != v2->size)
                return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

        __thrust_vector_mul<T>(v1, v2);
        return OK_STATUS_CUDA;
}

template<typename T>
ok_status vector_div(vector_<T> *v1, const vector_<T> *v2)
{
        OK_CHECK_VECTOR(v1);
        OK_CHECK_VECTOR(v2);
        if (v1->size != v2->size)
                return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

        __thrust_vector_div<T>(v1, v2);
        return OK_STATUS_CUDA;
}

template<typename T>
ok_status vector_add_constant_(vector_<T> *v, const T x)
{
        OK_CHECK_VECTOR(v);
        __thrust_vector_add_constant<T>(v, x);
        return OK_STATUS_CUDA;
}


template<typename T>
ok_status vector_indmin_(const vector_<T> *v, const T default_value,
	size_t *idx)
{
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(idx);
	*idx = __thrust_vector_indmin<T>(v);
	return OK_STATUS_CUDA;

}

template<typename T>
ok_status vector_min_(const vector_<T> *v, const T default_value, T *minval)
{
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(minval);
	*minval = __thrust_vector_min<T>(v);
	return OK_STATUS_CUDA;
}

template<typename T>
ok_status vector_max_(const vector_<T> *v, const T default_value, T *maxval)
{
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(maxval);
	*maxval = __thrust_vector_max<T>(v);
	return OK_STATUS_CUDA;
}

/* explicit instantiations for downstream code*/
template ok_status vector_alloc_(vector_<ok_float> *, size_t);
template ok_status vector_calloc_(vector_<ok_float> *, size_t);
template ok_status vector_free_(vector_<ok_float> *);
template ok_status vector_set_all_(vector_<ok_float> *, ok_float);
template ok_status vector_memcpy_vv_(vector_<ok_float> *, const vector_<ok_float> *);
template ok_status vector_memcpy_av_(ok_float *, const vector_<ok_float> *, size_t);
template ok_status vector_memcpy_va_(vector_<ok_float> *, const ok_float *, size_t);
template ok_status vector_scale_(vector_<ok_float> *, ok_float);
template ok_status vector_add_(vector_<ok_float> *, const vector_<ok_float> *);
template ok_status vector_sub_(vector_<ok_float> *, const vector_<ok_float> *);
template ok_status vector_mul_(vector_<ok_float> *, const vector_<ok_float> *);
template ok_status vector_div_(vector_<ok_float> *, const vector_<ok_float> *);
template ok_status vector_add_constant_(vector_<ok_float> *, const ok_float);
template ok_status vector_indmin_(const vector_<ok_float> *, const ok_float, size_t *);
template ok_status vector_min_(const vector_<ok_float> *, const ok_float, ok_float *);
template ok_status vector_max_(const vector_<ok_float> *, const ok_float, ok_float *);

template ok_status vector_alloc_(vector_<size_t> *, size_t);
template ok_status vector_calloc_(vector_<size_t> *, size_t);
template ok_status vector_free_(vector_<size_t> *);
template ok_status vector_set_all_(vector_<size_t> *, size_t);
template ok_status vector_memcpy_vv_(vector_<size_t> *, const vector_<size_t> *);
template ok_status vector_memcpy_av_(size_t *, const vector_<size_t> *, size_t);
template ok_status vector_memcpy_va_(vector_<size_t> *, const size_t *, size_t);
template ok_status vector_indmin_(const vector_<size_t> *, const size_t, size_t *);
template ok_status vector_min_(const vector_<size_t> *, const size_t, size_t *);
template ok_status vector_max_(const vector_<size_t> *, const size_t, size_t *);

template ok_status vector_alloc_(vector_<ok_int> *, size_t);
template ok_status vector_calloc_(vector_<ok_int> *, size_t);
template ok_status vector_free_(vector_<ok_int> *);
template ok_status vector_set_all_(vector_<ok_int> *, ok_int);
template ok_status vector_memcpy_vv_(vector_<ok_int> *, const vector_<ok_int> *);
template ok_status vector_memcpy_av_(ok_int *, const vector_<ok_int> *, size_t);
template ok_status vector_memcpy_va_(vector_<ok_int> *, const ok_int *, size_t);

#ifdef __cplusplus
extern "C" {
#endif

/* VECTOR */
ok_status vector_print(const vector *v)
{
	uint i;
	ok_float v_host[v->size];
	OK_RETURNIF_ERR( vector_memcpy_av(v_host, v, 1) );
	for (i = 0; i < v->size; ++i)
		printf("%e ", v_host[i]);
	printf("\n");
	return OPTKIT_SUCCESS;
}

ok_status vector_scale(vector *v, ok_float x)
        { return vector_scale_<ok_float>(v, x); }

ok_status vector_abs(vector *v)
{
	OK_CHECK_VECTOR(v);
	__thrust_vector_abs(v);
	return OK_STATUS_CUDA;
}

ok_status vector_recip(vector *v)
{
	OK_CHECK_VECTOR(v);
	__thrust_vector_recip(v);
	return OK_STATUS_CUDA;
}

ok_status vector_safe_recip(vector *v)
{
	OK_CHECK_VECTOR(v);
	__thrust_vector_safe_recip(v);
	return OK_STATUS_CUDA;
}

ok_status vector_sqrt(vector *v)
{
	OK_CHECK_VECTOR(v);
	__thrust_vector_sqrt(v);
	return OK_STATUS_CUDA;
}

ok_status vector_pow(vector *v, const ok_float x)
{
	OK_CHECK_VECTOR(v);
	__thrust_vector_pow(v, x);
	return OK_STATUS_CUDA;
}

ok_status vector_exp(vector *v)
{
	OK_CHECK_VECTOR(v);
	__thrust_vector_exp(v);
	return OK_STATUS_CUDA;
}

ok_status vector_uniform_rand(vector *v, const ok_float minval,
	const ok_float maxval)
{
	OK_RETURNIF_ERR( ok_rand_u01(v->data, v->size, v->stride) );
	OK_RETURNIF_ERR( vector_scale(v, maxval - minval) );
	return OK_SCAN_ERR( vector_add_constant(v, minval) );
}

/* INDVECTOR */
ok_status indvector_print(const indvector *v)
{
	uint i;
	size_t v_host[v->size];
	OK_RETURNIF_ERR( indvector_memcpy_av(v_host, v, 1) );
	for (i = 0; i < v->size; ++i)
		printf("%zu ", v_host[i]);
	printf("\n");
	return OPTKIT_SUCCESS;
}

/* INT_VECTOR */
ok_status int_vector_print(const int_vector *v)
{
        uint i;
        int v_host[v->size];
        OK_RETURNIF_ERR( int_vector_memcpy_av(v_host, v, 1) );
        for (i = 0; i < v->size; ++i)
                printf("%i ", v_host[i]);
        printf("\n");
        return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
