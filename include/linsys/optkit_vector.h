#ifndef OPTKIT_LINSYS_VECTOR_H_
#define OPTKIT_LINSYS_VECTOR_H_

#include "optkit_defs.h"

#ifdef __cplusplus
template<typename T>
struct vector_ {
public:
	size_t size, stride;
	T * data;
};

template<typename T>
void vector_alloc_(vector_<T> * v, size_t n);
template<typename T>
void vector_calloc_(vector_<T> * v, size_t n);
template<typename T>
void vector_free_(vector_<T> * v);
template<typename T>
void vector_set_all_(vector_<T> * v, T x);
template<typename T>
void vector_subvector_(vector_<T> * v_out, vector_<T> * v_in, size_t offset,
	size_t n);
template<typename T>
void vector_view_array_(vector_<T> * v, T * base, size_t n);
template<typename T>
void vector_memcpy_vv_(vector_<T> * v_out, const vector_<T> * v_in);
template<typename T>
void vector_memcpy_va_(vector_<T> * v_out, const T * arr_in, size_t stride_arr);
template<typename T>
void vector_memcpy_av_(T * arr_out, const vector_<T> * v_in, size_t stride_arr);
template<typename T>
void vector_scale_(vector_<T> * v, T x);
template<typename T>
void vector_add_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
void vector_sub_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
void vector_mul_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
void vector_div_(vector_<T> * v1, const vector_<T> * v2);
template<typename T>
void vector_add_constant_(vector_<T> *v, const T x);
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

void vector_alloc(vector * v, size_t n);
void vector_calloc(vector * v, size_t n);
void vector_free(vector * v);
void vector_set_all(vector * v, ok_float x);
void vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n);
void vector_view_array(vector * v, ok_float * base, size_t n);
void vector_memcpy_vv(vector * v1, const vector * v2);
void vector_memcpy_va(vector * v, const ok_float * y, size_t stride_y);
void vector_memcpy_av(ok_float * x, const vector * v, size_t stride_x);
void vector_print(const vector *v);
void vector_scale(vector * v, ok_float x);
void vector_add(vector * v1, const vector * v2);
void vector_sub(vector * v1, const vector * v2);
void vector_mul(vector * v1, const vector * v2);
void vector_div(vector * v1, const vector * v2);
void vector_add_constant(vector *v, const ok_float x);
void vector_abs(vector * v);
void vector_recip(vector * v);
void vector_safe_recip(vector * v);
void vector_sqrt(vector * v);
void vector_pow(vector * v, const ok_float x);
void vector_exp(vector * v);
size_t vector_indmin(vector * v);
ok_float vector_min(vector * v);
ok_float vector_max(vector * v);

#ifdef __cplusplus
typedef vector_<size_t> indvector;
#else
typedef struct indvector {
	size_t size, stride;
	size_t * data;
} indvector;
#endif

void indvector_alloc(indvector * v, size_t n);
void indvector_calloc(indvector * v, size_t n);
void indvector_free(indvector * v);
void indvector_set_all(indvector * v, size_t x);
void indvector_subvector(indvector * v_out, indvector * v_in, size_t offset,
	size_t n);
void indvector_view_array(indvector * v, size_t * base, size_t n);
void indvector_memcpy_vv(indvector * v1, const indvector * v2);
void indvector_memcpy_va(indvector * v, const size_t * y, size_t stride_y);
void indvector_memcpy_av(size_t * x, const indvector * v, size_t stride_x);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_LINSYS_VECTOR_H */
