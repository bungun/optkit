#ifndef OPTKIT_LINSYS_MATRIX_H_
#define OPTKIT_LINSYS_MATRIX_H_

#include "optkit_vector.hpp"

template<typename T>
struct matrix_ {
	size_t size1, size2, ld;
	T * data;
	enum CBLAS_ORDER order;
};

template<typename T>
void matrix_alloc_(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord);
template<typename T>
void matrix_calloc_(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord);
template<typename T>
void matrix_free_(matrix_<T> * A);
template<typename T>
void matrix_submatrix_(matrix_<T> * A_sub, matrix_<T> * A, size_t i, size_t j,
	size_t n1, size_t n2);
template<typename T>
void matrix_row_(vector * row, matrix_<T> * A, size_t i);
template<typename T>
void matrix_column_(vector * col, matrix_<T> * A, size_t j);
template<typename T>
void matrix_diagonal_(vector * diag, matrix_<T> * A);
template<typename T>
void matrix_cast_vector_(vector * v, matrix_<T> * A);
template<typename T>
void matrix_view_array_(matrix_<T> * A, const T * base, size_t n1, size_t n2,
	enum CBLAS_ORDER ord);
template<typename T>
void matrix_set_all_(matrix_<T> * A, T x);
template<typename T>
void matrix_memcpy_mm_(matrix_<T> * A, const matrix_<T> *B);
template<typename T>
void matrix_memcpy_ma_(matrix_<T> * A, const T *B,
	const enum CBLAS_ORDER ord);
template<typename T>
void matrix_memcpy_am_(T * A, const matrix_<T> *B,
	const enum CBLAS_ORDER ord);
template<typename T>
void matrix_scale_(matrix_<T> * A, T x);
template<typename T>
void matrix_scale_left_(matrix_<T> * A, const vector * v);
template<typename T>
void matrix_scale_right_(matrix_<T> * A, const vector * v);

#ifdef __cplusplus
extern "C" {
#endif

typedef matrix_<ok_float> matrix;

void matrix_alloc_(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord);
void matrix_calloc(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord);
void matrix_free(matrix_<T> * A);
void matrix_submatrix(matrix_<T> * A_sub, matrix_<T> * A, size_t i, size_t j,
	size_t n1, size_t n2);
void matrix_row(vector * row, matrix_<T> * A, size_t i);
void matrix_column(vector * col, matrix_<T> * A, size_t j);
void matrix_diagonal(vector * diag, matrix_<T> * A);
void matrix_cast_vector(vector * v, matrix_<T> * A);
void matrix_view_array(matrix_<T> * A, const ok_float * base, size_t n1, size_t n2,
	enum CBLAS_ORDER ord);
void matrix_set_all(matrix_<T> * A, ok_float x);
void matrix_memcpy_mm(matrix_<T> * A, const matrix_<T> *B);
void matrix_memcpy_ma(matrix_<T> * A, const ok_float *B,
	const enum CBLAS_ORDER ord);
void matrix_memcpy_am(ok_float * A, const matrix_<T> *B,
	const enum CBLAS_ORDER ord);
void matrix_print(matrix_<T> * A);
void matrix_scale(matrix_<T> * A, ok_float x);
void matrix_scale_left(matrix_<T> * A, const vector * v);
void matrix_scale_right(matrix_<T> * A, const vector * v);
void matrix_abs(matrix_<T> * A);
void matrix_pow(matrix_<T> * A, const ok_float p);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_LINSYS_MATRIX_H_ */
