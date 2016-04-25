#ifndef OPTKIT_LINSYS_MATRIX_H_
#define OPTKIT_LINSYS_MATRIX_H_

#include "optkit_vector.h"

#ifndef OK_CHECK_MATRIX
#define OK_CHECK_MATRIX(M) if (!M || !M->data) return OPTKIT_ERROR_UNALLOCATED;
#endif

#ifdef __cplusplus
template<typename T>
struct matrix_ {
	size_t size1, size2, ld;
	T * data;
	enum CBLAS_ORDER order;
};

template<typename T>
ok_status matrix_alloc_(matrix_<T> * A, size_t m, size_t n,
	enum CBLAS_ORDER ord);
template<typename T>
ok_status matrix_calloc_(matrix_<T> * A, size_t m, size_t n,
	enum CBLAS_ORDER ord);
template<typename T>
ok_status matrix_free_(matrix_<T> * A);
template<typename T>
ok_status matrix_submatrix_(matrix_<T> * A_sub, matrix_<T> * A, size_t i,
	size_t j, size_t n1, size_t n2);
template<typename T>
ok_status matrix_row_(vector_<T> * row, matrix_<T> * A, size_t i);
template<typename T>
ok_status matrix_column_(vector_<T> * col, matrix_<T> * A, size_t j);
template<typename T>
ok_status matrix_diagonal_(vector_<T> * diag, matrix_<T> * A);
template<typename T>
ok_status matrix_cast_vector_(vector_<T> * v, matrix_<T> * A);
template<typename T>
ok_status matrix_view_array_(matrix_<T> * A, const T * base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord);
template<typename T>
ok_status matrix_set_all_(matrix_<T> * A, T x);
template<typename T>
ok_status matrix_memcpy_mm_(matrix_<T> * A, const matrix_<T> * B);
template<typename T>
ok_status matrix_memcpy_ma_(matrix_<T> * A, const T * B,
	const enum CBLAS_ORDER ord);
template<typename T>
ok_status matrix_memcpy_am_(T * A, const matrix_<T> * B,
	const enum CBLAS_ORDER ord);
template<typename T>
ok_status matrix_scale_(matrix_<T> * A, T x);
template<typename T>
ok_status matrix_scale_left_(matrix_<T> * A, const vector_<T> * v);
template<typename T>
ok_status matrix_scale_right_(matrix_<T> * A, const vector_<T> * v);
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
typedef matrix_<ok_float> matrix;
#else
typedef struct matrix {
	size_t size1, size2, ld;
	ok_float * data;
	enum CBLAS_ORDER order;
} matrix;
#endif

ok_status matrix_alloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord);
ok_status matrix_calloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord);
ok_status matrix_free(matrix * A);
ok_status matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j,
	size_t n1, size_t n2);
ok_status matrix_row(vector * row, matrix * A, size_t i);
ok_status matrix_column(vector * col, matrix * A, size_t j);
ok_status matrix_diagonal(vector * diag, matrix * A);
ok_status matrix_cast_vector(vector * v, matrix * A);
ok_status matrix_view_array(matrix * A, const ok_float * base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord);
ok_status matrix_set_all(matrix * A, ok_float x);
ok_status matrix_memcpy_mm(matrix * A, const matrix *B);
ok_status matrix_memcpy_ma(matrix * A, const ok_float *B,
	const enum CBLAS_ORDER ord);
ok_status matrix_memcpy_am(ok_float * A, const matrix *B,
	const enum CBLAS_ORDER ord);
ok_status matrix_print(matrix * A);
ok_status matrix_scale(matrix * A, ok_float x);
ok_status matrix_scale_left(matrix * A, const vector * v);
ok_status matrix_scale_right(matrix * A, const vector * v);
ok_status matrix_abs(matrix * A);
ok_status matrix_pow(matrix * A, const ok_float p);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_LINSYS_MATRIX_H_ */
