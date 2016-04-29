#include "optkit_vector.h"

template<typename T>
ok_status vector_alloc_(vector_<T> * v, size_t n)
{
	OK_CHECK_PTR(v);
	if (v->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	v->size = n;
	v->stride = 1;
	v->data = (T*) malloc(n * sizeof(T));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_calloc_(vector_<T> * v, size_t n)
{
	OK_RETURNIF_ERR( vector_alloc_<T>(v, n) );
	memset(v->data, 0, n * sizeof(T));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_free_(vector_<T> * v)
{
	OK_CHECK_VECTOR(v);
	ok_free(v->data);
	v->size = (size_t) 0;
	v->stride = (size_t) 0;
	return OPTKIT_SUCCESS;
}

template<typename T>
static inline void __vector_set(vector_<T> * v, size_t i, T x)
{
        v->data[i * v->stride] = x;
}

template<typename T>
static inline T __vector_get(const vector_<T> *v, size_t i)
{
	return v->data[i * v->stride];
}

template<typename T>
ok_status vector_set_all_(vector_<T> * v, T x)
{
	uint i;
	OK_CHECK_VECTOR(v);
	for (i = 0; i < v->size; ++i)
		__vector_set<T>(v, i, x);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_subvector_(vector_<T> * v_out, vector_<T> * v_in,
	size_t offset, size_t n)
{
	if (!v_out || !v_in || !v_in->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	v_out->size = n;
	v_out->stride = v_in->stride;
	v_out->data = v_in->data + offset * v_in->stride;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_view_array_(vector_<T> * v, T * base, size_t n)
{
	if (!v || !base)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	v->size = n;
	v->stride = 1;
	v->data = base;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_memcpy_vv_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	OK_CHECK_VECTOR(v1);
	OK_CHECK_VECTOR(v2);
	if (v1->size != v2->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if ( v1->stride == 1 && v2->stride == 1)
		memcpy(v1->data, v2->data, v1->size * sizeof(T));
	else
		// #ifdef _OPENMP
		// #pragma omp parallel for
		// #endif
		for (i = 0; i < v1->size; ++i)
			__vector_set<T>(v1, i, __vector_get<T>(v2,i));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_memcpy_va_(vector_<T> * v, const T * y, size_t stride_y)
{
	uint i;
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(y);

	if (v->stride == 1 && stride_y == 1)
		memcpy(v->data, y, v->size * sizeof(T));
	else
		// #ifdef _OPENMP
		// #pragma omp parallel for
		// #endif
		for (i = 0; i < v->size; ++i)
			__vector_set<T>(v, i, y[i * stride_y]);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_memcpy_av_(T * x, const vector_<T> * v, size_t stride_x)
{
	uint i;
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(x);

	if (v->stride == 1 && stride_x == 1)
		memcpy(x, v->data, v->size * sizeof(T));
	else
		// #ifdef _OPENMP
		// #pragma omp parallel for
		// #endif
		for (i = 0; i < v->size; ++i)
			x[i * stride_x] = __vector_get<T>(v,i);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_scale_(vector_<T> * v, T x)
{
	uint i;
	OK_CHECK_VECTOR(v);

	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] *= x;
	return OPTKIT_SUCCESS;
}


template<typename T>
ok_status vector_add_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	OK_CHECK_VECTOR(v1);
	OK_CHECK_VECTOR(v2);
	if (v1->size != v2->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] += v2->data[i * v2->stride];
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_sub_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	OK_CHECK_VECTOR(v1);
	OK_CHECK_VECTOR(v2);
	if (v1->size != v2->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] -= v2->data[i * v2->stride];
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_mul_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	OK_CHECK_VECTOR(v1);
	OK_CHECK_VECTOR(v2);
	if (v1->size != v2->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] *= v2->data[i * v2->stride];
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_div_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	OK_CHECK_VECTOR(v1);
	OK_CHECK_VECTOR(v2);
	if (v1->size != v2->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] /= v2->data[i * v2->stride];
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_add_constant_(vector_<T> * v, const T x)
{
	uint i;
	OK_CHECK_VECTOR(v);

	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] += x;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_indmin_(const vector_<T> * v, const T default_value,
	size_t * idx)
{
	T minval = default_value;
	size_t minind = 0;
	size_t i;
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(idx);

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : minval, minind)
	#endif
	for (i = 0; i < v->size; ++i) {
		if(v->data[i * v->stride] < minval) {
			minval = v->data[i * v->stride];
			minind = i;
		}
	}
	*idx = minind;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_min_(const vector_<T> * v, const T default_value, T * minval)
{
	T minval_ = default_value;
	size_t i;
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(minval);

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : minval_)
	#endif
	for(i = 0; i < v->size; ++i)
		if(v->data[i * v->stride] < minval_)
			minval_ = v->data[i * v->stride];

	*minval = minval_;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_max_(const vector_<T> * v, const T default_value, T * maxval)
{
	T maxval_ = default_value;
	size_t i;
	OK_CHECK_VECTOR(v);
	OK_CHECK_PTR(maxval);

	#ifdef _OPENMP
	#pragma omp parallel for reduction(max : maxval_)
	#endif
	for(i = 0; i < v->size; ++i)
		if(v->data[i * v->stride] > maxval_)
			maxval_ = v->data[i * v->stride];

	*maxval = maxval_;
	return OPTKIT_SUCCESS;
}

/* explicit instantiations for downstream code*/
/* vector_scale required by: equilibration */
template ok_status vector_scale_(vector_<float> * v, float x);
template ok_status vector_scale_(vector_<double> * v, double x);

#ifdef __cplusplus
extern "C" {
#endif

ok_status vector_alloc(vector * v, size_t n)
	{ return vector_alloc_<ok_float>(v, n); }

ok_status vector_calloc(vector * v, size_t n)
	{ return vector_calloc_<ok_float>(v, n); }

ok_status vector_free(vector * v)
	{ return vector_free_<ok_float>(v); }

ok_status vector_set_all(vector * v, ok_float x)
	{ return vector_set_all_<ok_float>(v, x); }

ok_status vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n)
	{ return vector_subvector_<ok_float>(v_out, v_in, offset, n); }

ok_status vector_view_array(vector * v, ok_float * base, size_t n)
	{ return vector_view_array_<ok_float>(v, base, n); }

ok_status vector_memcpy_vv(vector * v1, const vector * v2)
	{ return vector_memcpy_vv_<ok_float>(v1, v2); }

ok_status vector_memcpy_va(vector * v, const ok_float *y, size_t stride_y)
	{ return vector_memcpy_va_<ok_float>(v, y, stride_y); }

ok_status vector_memcpy_av(ok_float * x, const vector * v, size_t stride_x)
	{ return vector_memcpy_av_<ok_float>(x, v, stride_x); }

ok_status vector_print(const vector * v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	for (i = 0; i < v->size; ++i)
		printf("%e ", __vector_get<ok_float>(v, i));
	printf("\n");
	return OPTKIT_SUCCESS;
}

ok_status vector_scale(vector * v, ok_float x)
{
	OK_CHECK_VECTOR(v);
	CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
	return OPTKIT_SUCCESS;
}

ok_status vector_add(vector * v1, const vector * v2)
	{ return vector_add_<ok_float>(v1, v2); }

ok_status vector_sub(vector * v1, const vector * v2)
	{ return vector_sub_<ok_float>(v1, v2); }

ok_status vector_mul(vector * v1, const vector * v2)
	{ return vector_mul_<ok_float>(v1, v2); }

ok_status vector_div(vector * v1, const vector * v2)
	{ return vector_div_<ok_float>(v1, v2); }

ok_status vector_add_constant(vector * v, const ok_float x)
	{ return vector_add_constant_<ok_float>(v, x); }

ok_status vector_abs(vector * v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(fabs)(v->data[i * v->stride]);
	return OPTKIT_SUCCESS;
}

ok_status vector_recip(vector * v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = kOne / v->data[i * v->stride];
	return OPTKIT_SUCCESS;
}

ok_status vector_safe_recip(vector * v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		if (v->data[i * v->stride] > 0)
			v->data[i * v->stride] = kOne / v->data[i * v->stride];
	return OPTKIT_SUCCESS;
}

ok_status vector_sqrt(vector * v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(sqrt)(v->data[i * v->stride]);
	return OPTKIT_SUCCESS;
}

ok_status vector_pow(vector * v, const ok_float x)
{
	uint i;
	OK_CHECK_VECTOR(v);
	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(pow)(v->data[i * v->stride], x);
	return OPTKIT_SUCCESS;
}

ok_status vector_exp(vector * v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	// #ifdef _OPENMP
	// #pragma omp parallel for
	// #endif
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(exp)(v->data[i * v->stride]);
	return OPTKIT_SUCCESS;
}

ok_status vector_indmin(const vector * v, size_t * idx)
	{ return vector_indmin_<ok_float>(v, OK_FLOAT_MAX, idx); }

ok_status vector_min(const vector * v, ok_float * minval)
	{ return vector_min_<ok_float>(v, OK_FLOAT_MAX, minval); }

ok_status vector_max(const vector * v, ok_float * maxval)
	{ return vector_max_<ok_float>(v, -OK_FLOAT_MAX, maxval); }

ok_status indvector_alloc(indvector * v, size_t n)
	{ return vector_alloc_<size_t>(v, n); }

ok_status indvector_calloc(indvector * v, size_t n)
	{ return vector_calloc_<size_t>(v, n); }

ok_status indvector_free(indvector * v)
	{ return vector_free_<size_t>(v); }

ok_status indvector_set_all(indvector * v, size_t x)
	{ return vector_set_all_<size_t>(v, x); }

ok_status indvector_subvector(indvector * v_out, indvector * v_in,
	size_t offset, size_t n)
	{ return vector_subvector_<size_t>(v_out, v_in, offset, n); }

ok_status indvector_view_array(indvector * v, size_t * base, size_t n)
	{ return vector_view_array_<size_t>(v, base, n); }

ok_status indvector_memcpy_vv(indvector * v1, const indvector * v2)
	{ return vector_memcpy_vv_<size_t>(v1, v2); }

ok_status indvector_memcpy_va(indvector * v, const size_t * y, size_t stride_y)
	{ return vector_memcpy_va_<size_t>(v, y, stride_y); }

ok_status indvector_memcpy_av(size_t * x, const indvector * v, size_t stride_x)
	{ return vector_memcpy_av_<size_t>(x, v, stride_x); }

ok_status indvector_print(const indvector * v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	for (i = 0; i < v->size; ++i)
		printf("%zu ", __vector_get<size_t>(v, i));
	printf("\n");
	return OPTKIT_SUCCESS;
}

ok_status indvector_indmin(const indvector * v, size_t * idx)
	{ return vector_indmin_<size_t>(v, (size_t) INT_MAX, idx); }

ok_status indvector_min(const indvector * v, size_t * minval)
	{ return vector_min_<size_t>(v, (size_t) INT_MAX, minval); }

ok_status indvector_max(const indvector * v, size_t * maxval)
	{ return vector_max_<size_t>(v, 0, maxval); }

#ifdef __cplusplus
}
#endif
