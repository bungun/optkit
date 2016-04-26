#ifndef OPTKIT_LINSYS_VECTOR_H_
#define OPTKIT_LINSYS_VECTOR_H_

#include "optkit_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OK_CHECK_VECTOR
#define OK_CHECK_VECTOR(v) \
	do { \
		if (!v || !v->data) \
			return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED ); \
	} while(0)
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
template<typename T>
struct vector_ {
public:
	size_t size, stride;
	T * data;
};

template<typename T>
ok_status vector_alloc_(vector_<T> * v, size_t n);
template<typename T>
ok_status vector_calloc_(vector_<T> * v, size_t n);
template<typename T>
ok_status vector_free_(vector_<T> * v);
template<typename T>
ok_status vector_set_all_(vector_<T> * v, T x);
template<typename T>
ok_status vector_subvector_(vector_<T> * v_out, vector_<T> * v_in,
	size_t offset, size_t n);
template<typename T>
ok_status vector_view_array_(vector_<T> * v, T * base, size_t n);
template<typename T>
ok_status vector_memcpy_vv_(vector_<T> * v_out, const vector_<T> * v_in);
template<typename T>
ok_status vector_memcpy_va_(vector_<T> * v_out, const T * arr_in,
	size_t stride_arr);
template<typename T>
ok_status vector_memcpy_av_(T * arr_out, const vector_<T> * v_in,
	size_t stride_arr);
template<typename T>
ok_status vector_scale_(vector_<T> * v, T x);
template<typename T>
ok_status vector_add_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
ok_status vector_sub_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
ok_status vector_mul_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
ok_status vector_div_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
ok_status vector_add_constant_(vector_<T> *v, const T x);
template<typename T>
ok_status vector_indmin_(const vector_<T> * v, const T default_value,
	size_t * idx);
template<typename T>
ok_status vector_min_(const vector_<T> * v, const T default_value, T * minval);
template<typename T>
ok_status vector_max_(const vector_<T> * v, const T default_value, T * maxval);
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
typedef vector_<ok_float> vector;
#else
typedef struct vector {
	size_t size, stride;
	ok_float * data;
} vector;
#endif

ok_status vector_alloc(vector * v, size_t n);
ok_status vector_calloc(vector * v, size_t n);
ok_status vector_free(vector * v);
ok_status vector_set_all(vector * v, ok_float x);
ok_status vector_subvector(vector * v_out, vector * v_in, size_t offset,
	size_t n);
ok_status vector_view_array(vector * v, ok_float * base, size_t n);
ok_status vector_memcpy_vv(vector * v1, const vector * v2);
ok_status vector_memcpy_va(vector * v, const ok_float * y, size_t stride_y);
ok_status vector_memcpy_av(ok_float * x, const vector * v, size_t stride_x);
ok_status vector_print(const vector *v);
ok_status vector_scale(vector * v, ok_float x);
ok_status vector_add(vector * v1, const vector * v2);
ok_status vector_sub(vector * v1, const vector * v2);
ok_status vector_mul(vector * v1, const vector * v2);
ok_status vector_div(vector * v1, const vector * v2);
ok_status vector_add_constant(vector *v, const ok_float x);
ok_status vector_abs(vector * v);
ok_status vector_recip(vector * v);
ok_status vector_safe_recip(vector * v);
ok_status vector_sqrt(vector * v);
ok_status vector_pow(vector * v, const ok_float x);
ok_status vector_exp(vector * v);
ok_status vector_indmin(const vector * v, size_t * idx);
ok_status vector_min(const vector * v, ok_float * minval);
ok_status vector_max(const vector * v, ok_float * maxval);

#ifdef __cplusplus
typedef vector_<size_t> indvector;
#else
typedef struct indvector {
	size_t size, stride;
	size_t * data;
} indvector;
#endif

ok_status indvector_alloc(indvector * v, size_t n);
ok_status indvector_calloc(indvector * v, size_t n);
ok_status indvector_free(indvector * v);
ok_status indvector_set_all(indvector * v, size_t x);
ok_status indvector_subvector(indvector * v_out, indvector * v_in,
	size_t offset, size_t n);
ok_status indvector_view_array(indvector * v, size_t * base, size_t n);
ok_status indvector_memcpy_vv(indvector * v1, const indvector * v2);
ok_status indvector_memcpy_va(indvector * v, const size_t * y, size_t stride_y);
ok_status indvector_memcpy_av(size_t * x, const indvector * v, size_t stride_x);
ok_status indvector_print(const indvector * v);
ok_status indvector_indmin(const indvector * v, size_t * idx);
ok_status indvector_min(const indvector * v, size_t * minval);
ok_status indvector_max(const indvector * v, size_t * maxval);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_LINSYS_VECTOR_H */
