#include "optkit_vector.hpp"

template<typename T>
static inline int __vector_exists(vector_<T> * v)
{
	if (v == OK_NULL) {
		printf("%s%s\n", "Error: cannot write to uninitialized ",
			"vector_<T> pointer\n");
		return 0;
	} else {
		return 1;
	}
}

template<typename T>
void vector_alloc_(vector_<T> * v, size_t n)
{
	if (!__vector_exists(v))
		return;
	v->size = n;
	v->stride = 1;
	v->data = (T*) malloc(n * sizeof(T));
}

template<typename T>
void vector_calloc_(vector_<T> * v, size_t n)
{
	vector_alloc(v, n);
	memset(v->data, 0, n * sizeof(T));
}

template<typename T>
void vector_free_(vector_<T> * v)
{
	if (v->data != OK_NULL)
		ok_free(v->data);
	v->size = (size_t) 0;
	v->stride = (size_t) 0;
}

template<typename T>
static inline void __vector_set(vector_<T> * v, size_t i, T x)
{
        v->data[i * v->stride] = x;
}

template<typename T>
static ok_float __vector_get(const vector_<T> *v, size_t i)
{
	return v->data[i * v->stride];
}

template<typename T>
void vector_set_all_(vector_<T> * v, ok_float x)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		__vector_set<T>(v, i, x);
}

template<typename T>
void vector_subvector_(vector_<T> * v_out, vector_<T> * v_in, size_t offset, size_t n)
{
	if (!__vector_exists(v_out))
		return;
	v_out->size = n;
	v_out->stride = v_in->stride;
	v_out->data = v_in->data + offset * v_in->stride;
}

template<typename T>
void vector_view_array_(vector_<T> * v, T * base, size_t n)
{
	if (!__vector_exists(v))
		return;
	v->size = n;
	v->stride = 1;
	v->data = base;
}

template<typename T>
void vector_memcpy_vv_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	if ( v1->stride == 1 && v2->stride == 1)
		memcpy(v1->data, v2->data, v1->size * sizeof(T));
	else
		for (i = 0; i < v1->size; ++i)
			__vector_set<T>(v1, i, __vector_get<T>(v2,i));
}

template<typename T>
void vector_memcpy_va_(vector_<T> * v, const T *y, size_t stride_y)
{
	uint i;
	if (v->stride == 1 && stride_y == 1)
		memcpy(v->data, y, v->size * sizeof(T));
	else
		for (i = 0; i < v->size; ++i)
			__vector_set<T>(v, i, y[i * stride_y]);
}

template<typename T>
void vector_memcpy_av_(T * x, const vector_<T> * v, size_t stride_x)
{
	uint i;
	if (v->stride == 1 && stride_x == 1)
		memcpy(x, v->data, v->size * sizeof(T));
	else
		for (i = 0; i < v->size; ++i)
			x[i * stride_x] = __vector_get<T>(v,i);
}

template<typename T>
void vector_add_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] += v2->data[i * v2->stride];
}

template<typename T>
void vector_sub_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] -= v2->data[i * v2->stride];
}

template<typename T>
void vector_mul_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] *= v2->data[i * v2->stride];
}

template<typename T>
void vector_div_(vector_<T> * v1, const vector_<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] /= v2->data[i * v2->stride];
}

template<typename T>
void vector_add_constant_(vector_<T> * v, const ok_float x)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] += x;
}

#ifdef __cplusplus
extern "C" {
#endif

void vector_alloc(vector * v, size_t n)
	{ vector_alloc_<ok_float>(v, n); }

void vector_calloc(vector * v, size_t n)
	{ vector_calloc_<ok_float>(v, n); }

void vector_free(vector * v)
	{ vector_free_<ok_float>(v, n); }

void vector_print(const vector * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		printf("%e ", __vector_get<ok_float>(v, i));
	printf("\n");
}

void vector_set_all_(vector * v, ok_float x)
	{ vector_set_all<ok_float>(v, x); }

void vector_subvector_(vector * v_out, vector * v_in, size_t offset, size_t n)
	{ vector_subvector<ok_float>(v_out, v_in, offset, n); }

void vector_view_array_(vector * v, T * base, size_t n)
	{ vector_view_array_<ok_float>(v, base, n); }

void vector_memcpy_vv_(vector * v1, const vector * v2)
	{ vector_memcpy_vv_<ok_float>(v1, v2); }

void vector_memcpy_va_(vector * v, const ok_float *y, size_t stride_y)
	{ vector_memcpy_va_<ok_float>(v, y, stride_y); }

void vector_memcpy_av_(ok_float * x, const vector * v, size_t stride_x)
	{ vector_memcpy_av_<ok_float>(x, v, stride_x); }

void vector_scale(vector * v, T x)
{
	CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
}

void vector_add(vector * v1, const vector * v2)
	{ vector_add_<ok_float>(v1, v2); }

void vector_sub(vector * v1, const vector * v2)
	{ vector_sub_<ok_float>(v1, v2); }

void vector_mul(vector * v1, const vector * v2)
	{ vector_mul_<ok_float>(v1, v2); }

void vector_div(vector * v1, const vector * v2)
	{ vector_div_<ok_float>(v1, v2); }

void vector_add_constant(vector * v, const ok_float x)
	{ vector_add_constant_<ok_float>(v, x); }

void vector_abs(vector * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(fabs)(v->data[i * v->stride]);
}

void vector_recip(vector * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = kOne / v->data[i * v->stride];
}

void vector_safe_recip(vector * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = ((ok_float)
			(v->data[i * v->stride] == kZero)) *
			kOne / v->data[i * v->stride];
}

void vector_sqrt(vector * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(sqrt)(v->data[i * v->stride]);
}

void vector_pow(vector * v, const ok_float x)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(pow)(v->data[i * v->stride], x);
}

void vector_exp(vector * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(exp)(v->data[i * v->stride], x);
}

size_t vector_indmin(vector * v)
{
	ok_float minval = OK_FLOAT_MAX;
	size_t minind = 0;
	size_t i;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : minval, minind)
	#endif
	for(i = 0; i < v->size; ++i)
	if(v->data[i] > minval) {
		minval = v->data[i];
		minind = i;
	}

	return i;
}

ok_float vector_min(vector * v)
{
	ok_float minval = OK_FLOAT_MAX;
	size_t i;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : minval)
	#endif
	for(i = 0; i < v->size; ++i)
	if(v->data[i] > minval)
		minval = v->data[i];

	return minval;
}

ok_float vector_max(vector * v)
{
	ok_float maxval = -OK_FLOAT_MAX;
	size_t i;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : maxval)
	#endif
	for(i = 0; i < v->size; ++i)
	if(v->data[i] > maxval)
		maxval = v->data[i];

	return maxval;
}

#ifdef __cplusplus
}
#endif
