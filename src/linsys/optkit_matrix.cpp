#include "optkit_matrix.h"

template<typename T>
void matrix_alloc_(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	A->size1 = m;
	A->size2 = n;
	A->data = (T *) malloc(m * n * sizeof(T));
	A->ld = (ord == CblasRowMajor) ? n : m;
	A->order = ord;
}

template<typename T>
void matrix_calloc_(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	if (!A)
		return;
	matrix_alloc_<T>(A, m, n, ord);
	memset(A->data, 0, m * n * sizeof(T));
}

template<typename T>
void matrix_free_(matrix_<T> * A)
{
	if (A && A->data)
		ok_free(A->data);
	A->size1 = (size_t) 0;
	A->size2 = (size_t) 0;
	A->ld = (size_t) 0;
}

template<typename T>
void matrix_submatrix_(matrix_<T> * A_sub, matrix_<T> * A,
    size_t i, size_t j, size_t n1, size_t n2)
{
	if (!A_sub || !A)
		return;
	A_sub->size1 = n1;
	A_sub->size2 = n2;
	A_sub->ld = A->ld;
	A_sub->data = (A->order == CblasRowMajor) ?
		      A->data + (i * A->ld) + j :
		      A->data + i + (j * A->ld);
	A_sub->order = A->order;
}

template<typename T>
void matrix_row_(vector_<T> * row, matrix_<T> * A, size_t i)
{
	if (!row || !A)
		return;
	row->size = A->size2;
	row->stride = (A->order == CblasRowMajor) ? 1 : A->ld;
	row->data = (A->order == CblasRowMajor) ?
		    A->data + (i * A->ld) :
		    A->data + i;
}

template<typename T>
void matrix_column_(vector_<T> * col, matrix_<T> *A, size_t j)
{
	if (!col || !A)
		return;
	col->size = A->size1;
	col->stride = (A->order == CblasRowMajor) ? A->ld : 1;
	col->data = (A->order == CblasRowMajor) ?
		    A->data + j :
		    A->data + (j * A->ld);
	}

template<typename T>
void matrix_diagonal_(vector_<T> * diag, matrix_<T> * A)
{
	if (!diag || !A)
		return;
	diag->data = A->data;
	diag->stride = A->ld + 1;
	diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

template<typename T>
void matrix_cast_vector_(vector_<T> * v, matrix_<T> * A)
{
	if (!v || !A)
		return;
	v->size = A->size1 * A->size2;
	v->stride = 1;
	v->data = A->data;
}

template<typename T>
void matrix_view_array_(matrix_<T> * A, const T *base, size_t n1, size_t n2,
        enum CBLAS_ORDER ord)
{
	if (!A)
		return;
	A->size1 = n1;
	A->size2 = n2;
	A->data = (T *) base;
	A->ld = (ord == CblasRowMajor) ? n2 : n1;
	A->order = ord;
}

template<typename T>
inline T __matrix_get_colmajor(const matrix_<T> * A, size_t i, size_t j)
{
	return A->data[i + j * A->ld];
}

template<typename T>
inline T __matrix_get_rowmajor(const matrix_<T> * A, size_t i, size_t j)
{
	return A->data[i * A->ld + j];
}

template<typename T>
inline void __matrix_set_rowmajor(matrix_<T> * A, size_t i, size_t j, T x)
{
	A->data[i * A->ld + j] = x;
}

template<typename T>
inline void __matrix_set_colmajor(matrix_<T> * A, size_t i, size_t j, T x)
{
	A->data[i + j * A->ld] = x;
}

template<typename T>
void matrix_set_all_(matrix_<T> * A, T x)
{
	size_t i, j;
	if (!A)
		return;

	if (A->order == CblasRowMajor)
		for (i = 0; i < A->size1; ++i)
			for (j = 0; j < A->size2; ++j)
				__matrix_set_rowmajor<T>(A, i, j, x);
	else
		for (j = 0; j < A->size2; ++j)
			for (i = 0; i < A->size1; ++i)
				__matrix_set_colmajor<T>(A, i, j, x);
}

template<typename T>
void matrix_memcpy_mm_(matrix_<T> * A, const matrix_<T> * B)
{
	uint i, j;
	if (A->size1 != B->size1) {
		printf("error: m-dimensions must match for matrix memcpy\n");
		return;
	} else if (A->size2 != B->size2) {
		printf("error: n-dimensions must match for matrix memcpy\n");
		return;
	}

	void (* mset)(matrix_<T> * M, size_t i, size_t j, T x) =
		(A->order == CblasRowMajor) ?
		__matrix_set_rowmajor<T> :
		__matrix_set_colmajor<T>;

	T (* mget)(const matrix_<T> * M, size_t i, size_t j) =
		(B->order == CblasRowMajor) ?
		__matrix_get_rowmajor<T> :
		__matrix_get_colmajor<T>;

	for (i = 0; i < A->size1; ++i)
		for (j = 0; j < A->size2; ++j)
			  mset(A, i, j, mget(B, i , j));
}

template<typename T>
void matrix_memcpy_ma_(matrix_<T> * A, const T * B, const enum CBLAS_ORDER ord)
{
	uint i, j;
	void (* mset)(matrix_<T> * M, size_t i, size_t j, T x) =
		(A->order == CblasRowMajor) ?
		__matrix_set_rowmajor<T> :
		__matrix_set_colmajor<T>;

	if (ord == CblasRowMajor)
		for (i = 0; i < A->size1; ++i)
			for (j = 0; j < A->size2; ++j)
				mset(A, i, j, B[i * A->size2 + j]);
	else
		for (i = 0; i < A->size1; ++i)
			for (j = 0; j < A->size2; ++j)
				mset(A, i, j, B[i + j * A->size1]);
}

template<typename T>
void matrix_memcpy_am_(T * A, const matrix_<T> * B, const enum CBLAS_ORDER ord)
{
	uint i, j;
	T (* mget)(const matrix_<T> * M, size_t i, size_t j) =
		(B->order == CblasRowMajor) ?
		__matrix_get_rowmajor<T> :
		__matrix_get_colmajor<T>;

	if (ord == CblasRowMajor)
		for (i = 0; i < B->size1; ++i)
			for (j = 0; j < B->size2; ++j)
				A[i * B->size2 + j] = mget(B, i, j);
	else
		for (j = 0; j < B->size2; ++j)
			for (i = 0; i < B->size1; ++i)
				A[i + B->size1 * j] = mget(B, i, j);
}

template<typename T>
void matrix_scale_(matrix_<T> *A, T x)
{
	size_t i;
	vector_<T> row_col;
	row_col.data = OK_NULL;
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row_<T>(&row_col, A, i);
			vector_scale_<T>(&row_col, x);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column_<T>(&row_col, A, i);
			vector_scale_<T>(&row_col, x);
		}
}

template<typename T>
void matrix_scale_left_(matrix_<T> * A, const vector_<T> * v)
{
	size_t i;
	vector_<T> row;
	row.data = OK_NULL;
	for(i = 0; i < A->size1; ++i) {
		matrix_row_<T>(&row, A, i);
		vector_scale_<T>(&row, v->data[i]);
	}
}

template<typename T>
void matrix_scale_right_(matrix_<T> * A, const vector_<T> * v)
{
	size_t i;
	vector_<T> col;
	col.data = OK_NULL;
	for(i = 0; i < A->size2; ++i) {
		matrix_column_<T>(&col, A, i);
		vector_scale_<T>(&col, v->data[i]);
	}
}

#ifdef __cplusplus
extern "C" {
#endif

void matrix_alloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ matrix_alloc_<ok_float>(A, m, n, ord); }

void matrix_calloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ matrix_calloc_<ok_float>(A, m, n, ord); }

void matrix_free(matrix * A)
	{ matrix_free_<ok_float>(A); }

void matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1,
	size_t n2)
	{ matrix_submatrix_<ok_float>(A_sub, A, i, j, n1, n2); }

void matrix_row(vector * row, matrix * A, size_t i)
	{ matrix_row_<ok_float>(row, A, i); }

void matrix_column(vector * col, matrix * A, size_t j)
	{ matrix_column_<ok_float>(col, A, j); }

void matrix_diagonal(vector * diag, matrix * A)
	{ matrix_diagonal_<ok_float>(diag, A); }

void matrix_cast_vector(vector * v, matrix * A)
	{ matrix_cast_vector_<ok_float>(v, A); }

void matrix_view_array(matrix * A, const ok_float * base, size_t n1, size_t n2,
        enum CBLAS_ORDER ord)
	{ matrix_view_array_<ok_float>(A, base, n1, n2, ord); }

void matrix_set_all(matrix * A, ok_float x)
	{ matrix_set_all_<ok_float>(A, x); }

void matrix_memcpy_mm(matrix * A, const matrix * B)
	{ matrix_memcpy_mm_<ok_float>(A, B); }

void matrix_memcpy_ma(matrix * A, const ok_float * B,
	const enum CBLAS_ORDER ord)
	{ matrix_memcpy_ma_<ok_float>(A, B, ord); }

void matrix_memcpy_am(ok_float * A, const matrix * B,
	const enum CBLAS_ORDER ord)
	{ matrix_memcpy_am_<ok_float>(A, B, ord); }

void matrix_print(matrix * A)
{
	uint i, j;
	ok_float (* mget)(const matrix * M, size_t i, size_t j) =
		(A->order == CblasRowMajor) ?
		__matrix_get_rowmajor<ok_float> :
		__matrix_get_colmajor<ok_float>;

	for (i = 0; i < A->size1; ++i) {
		for (j = 0; j < A->size2; ++j)
			printf("%e ", mget(A, i, j));
		printf("\n");
	}
	printf("\n");
}

void matrix_scale(matrix *A, ok_float x)
	{ matrix_scale_<ok_float>(A, x); }

void matrix_scale_left(matrix * A, const vector * v)
	{ matrix_scale_left_<ok_float>(A, v); }

void matrix_scale_right(matrix * A, const vector * v)
	{ matrix_scale_right_<ok_float>(A, v); }

void matrix_abs(matrix * A)
{
	size_t i;
	vector row_col;
	row_col.data = OK_NULL;
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			vector_abs(&row_col);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			vector_abs(&row_col);
		}
}

void matrix_pow(matrix * A, const ok_float x)
{
	size_t i;
	vector row_col;
	row_col.data = OK_NULL;
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			vector_pow(&row_col, x);
	}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			vector_pow(&row_col, x);
	}
}

#ifdef __cplusplus
}
#endif
