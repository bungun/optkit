#include <random>
#include "optkit_vector.h"

static ok_status ok_rand_u01(ok_float *x, const size_t size,
	const size_t stride)
{
	OK_CHECK_PTR(x);
	std::random_device rd;
	#ifdef FLOAT
	std::mt19937 generator(rd());
	#else
	std::mt19937_64 generator(rd());
	#endif
	std::uniform_real_distribution<ok_float> dist(kZero, kOne);
	uint i;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < size; ++i)
		x[i * stride] = dist(generator);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_alloc_(vector_<T> *v, size_t n)
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
ok_status vector_calloc_(vector_<T> *v, size_t n)
{
	OK_RETURNIF_ERR( vector_alloc_<T>(v, n) );
	memset(v->data, 0, n * sizeof(T));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_free_(vector_<T> *v)
{
	OK_CHECK_VECTOR(v);
	ok_free(v->data);
	v->size = (size_t) 0;
	v->stride = (size_t) 0;
	return OPTKIT_SUCCESS;
}

template<typename T>
static inline void __vector_set(vector_<T> *v, size_t i, T x)
{
        v->data[i * v->stride] = x;
}

template<typename T>
static inline T __vector_get(const vector_<T> *v, size_t i)
{
	return v->data[i * v->stride];
}

template<typename T>
ok_status vector_set_all_(vector_<T> *v, T x)
{
	uint i;
	OK_CHECK_VECTOR(v);
	for (i = 0; i < v->size; ++i)
		__vector_set<T>(v, i, x);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status vector_memcpy_vv_(vector_<T> *v1, const vector_<T> *v2)
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
ok_status vector_memcpy_va_(vector_<T> *v, const T *y, size_t stride_y)
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
ok_status vector_memcpy_av_(T *x, const vector_<T> *v, size_t stride_x)
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
ok_status vector_scale_(vector_<T> *v, T x)
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
ok_status vector_add_(vector_<T> *v1, const vector_<T> *v2)
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
ok_status vector_sub_(vector_<T> *v1, const vector_<T> *v2)
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
ok_status vector_mul_(vector_<T> *v1, const vector_<T> *v2)
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
ok_status vector_div_(vector_<T> *v1, const vector_<T> *v2)
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
ok_status vector_add_constant_(vector_<T> *v, const T x)
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
ok_status vector_indmin_(const vector_<T> *v, const T default_value,
	size_t *idx)
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
ok_status vector_min_(const vector_<T> *v, const T default_value, T *minval)
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
ok_status vector_max_(const vector_<T> *v, const T default_value, T *maxval)
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

ok_status vector_print(const vector *v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	for (i = 0; i < v->size; ++i)
		printf("%e ", __vector_get<ok_float>(v, i));
	printf("\n");
	return OPTKIT_SUCCESS;
}

ok_status vector_scale(vector *v, ok_float x)
{
	OK_CHECK_VECTOR(v);
	CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
	return OPTKIT_SUCCESS;
}

ok_status vector_abs(vector *v)
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

ok_status vector_recip(vector *v)
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

ok_status vector_safe_recip(vector *v)
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

ok_status vector_sqrt(vector *v)
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

ok_status vector_pow(vector *v, const ok_float x)
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

ok_status vector_exp(vector *v)
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


ok_status vector_uniform_rand(vector *v, const ok_float minval,
	const ok_float maxval)
{
	OK_RETURNIF_ERR( ok_rand_u01(v->data, v->size, v->stride) );
	OK_RETURNIF_ERR( vector_scale(v, maxval - minval) );
	return OK_SCAN_ERR( vector_add_constant(v, minval) );
}

ok_status indvector_print(const indvector *v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	for (i = 0; i < v->size; ++i)
		printf("%zu ", __vector_get<size_t>(v, i));
	printf("\n");
	return OPTKIT_SUCCESS;
}

/* INT_VECTOR */
ok_status int_vector_print(const int_vector *v)
{
	uint i;
	OK_CHECK_VECTOR(v);
	for (i = 0; i < v->size; ++i)
		printf("%i ", __vector_get<ok_int>(v, i));
	printf("\n");
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
