#include "optkit_matrix.h"

template<typename T>
ok_status matrix_alloc_(matrix_<T> *A, size_t m, size_t n,
	enum CBLAS_ORDER ord)
{
	OK_CHECK_PTR(A);
	if (A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	A->size1 = m;
	A->size2 = n;
	A->data = (T *) malloc(m * n * sizeof(T));
	A->ld = (ord == CblasRowMajor) ? n : m;
	A->order = ord;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_calloc_(matrix_<T> *A, size_t m, size_t n,
	enum CBLAS_ORDER ord)
{
	OK_RETURNIF_ERR( matrix_alloc_<T>(A, m, n, ord) );
	memset(A->data, 0, m * n * sizeof(T));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_free_(matrix_<T> *A)
{
	OK_CHECK_MATRIX(A);
	ok_free(A->data);
	A->size1 = (size_t) 0;
	A->size2 = (size_t) 0;
	A->ld = (size_t) 0;
	return OPTKIT_SUCCESS;
}

template<typename T>
inline T __matrix_get_colmajor(const matrix_<T> *A, size_t i, size_t j)
{
	return A->data[i + j * A->ld];
}

template<typename T>
inline T __matrix_get_rowmajor(const matrix_<T> *A, size_t i, size_t j)
{
	return A->data[i * A->ld + j];
}

template<typename T>
inline void __matrix_set_rowmajor(matrix_<T> *A, size_t i, size_t j, T x)
{
	A->data[i * A->ld + j] = x;
}

template<typename T>
inline void __matrix_set_colmajor(matrix_<T> *A, size_t i, size_t j, T x)
{
	A->data[i + j * A->ld] = x;
}

template<typename T>
ok_status matrix_set_all_(matrix_<T> *A, T x)
{
	size_t i, j;
	OK_CHECK_MATRIX(A);

	if (A->order == CblasRowMajor)
		for (i = 0; i < A->size1; ++i)
			for (j = 0; j < A->size2; ++j)
				__matrix_set_rowmajor<T>(A, i, j, x);
	else
		for (j = 0; j < A->size2; ++j)
			for (i = 0; i < A->size1; ++i)
				__matrix_set_colmajor<T>(A, i, j, x);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_memcpy_mm_(matrix_<T> *A, const matrix_<T> *B)
{
	uint i, j;
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(B);
	if (A->size1 != B->size1 || A->size2 != B->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	void (* mset)(matrix_<T> *M, size_t i, size_t j, T x) =
		(A->order == CblasRowMajor) ?
		__matrix_set_rowmajor<T> :
		__matrix_set_colmajor<T>;

	T (* mget)(const matrix_<T> *M, size_t i, size_t j) =
		(B->order == CblasRowMajor) ?
		__matrix_get_rowmajor<T> :
		__matrix_get_colmajor<T>;

	for (i = 0; i < A->size1; ++i)
		for (j = 0; j < A->size2; ++j)
			  mset(A, i, j, mget(B, i , j));
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_memcpy_ma_(matrix_<T> *A, const T *B,
	const enum CBLAS_ORDER ord)
{
	uint i, j;
	OK_CHECK_MATRIX(A);
	OK_CHECK_PTR(B);

	void (* mset)(matrix_<T> *M, size_t i, size_t j, T x) =
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
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_memcpy_am_(T *A, const matrix_<T> *B,
	const enum CBLAS_ORDER ord)
{
	uint i, j;
	OK_CHECK_MATRIX(B);
	OK_CHECK_PTR(A);

	T (* mget)(const matrix_<T> *M, size_t i, size_t j) =
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
	return OPTKIT_SUCCESS;
}


template<typename T>
ok_status matrix_scale_left_(matrix_<T> *A, const vector_<T> *v)
{
	size_t i;
	vector_<T> row;
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);

	if (A->size1 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	for(i = 0; i < A->size1; ++i) {
		row.data = OK_NULL;
		OK_CHECK_ERR(err, matrix_row_<T>(&row, A, i));
		OK_CHECK_ERR(err, vector_scale_<T>(&row, v->data[i]));
	}
	return err;
}

template<typename T>
ok_status matrix_scale_right_(matrix_<T> *A, const vector_<T> *v)
{
	size_t i;
	vector_<T> col;
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);
	if (A->size2 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	for(i = 0; i < A->size2; ++i) {
		col.data = OK_NULL;
		OK_CHECK_ERR(err, matrix_column_<T>(&col, A, i));
		OK_CHECK_ERR(err, vector_scale_<T>(&col, v->data[i]));
	}
	return OPTKIT_SUCCESS;
}

template ok_status matrix_alloc_(matrix_<ok_float> *, size_t, size_t, enum CBLAS_ORDER);
template ok_status matrix_calloc_(matrix_<ok_float> *, size_t, size_t, enum CBLAS_ORDER);
template ok_status matrix_free_(matrix_<ok_float> *);
template ok_status matrix_set_all_(matrix_<ok_float> *, ok_float);
template ok_status matrix_memcpy_mm_(matrix_<ok_float> *, const matrix_<ok_float> *);
template ok_status matrix_memcpy_am_(ok_float *, const matrix_<ok_float> *, enum CBLAS_ORDER);
template ok_status matrix_memcpy_ma_(matrix_<ok_float> *, const ok_float *, enum CBLAS_ORDER);
template ok_status matrix_scale_left_(matrix_<ok_float> *, const vector_<ok_float> *);
template ok_status matrix_scale_right_(matrix_<ok_float> *, const vector_<ok_float> *);

#ifdef __cplusplus
extern "C" {
#endif

ok_status matrix_print(matrix *A)
{
	uint i, j;
	ok_float (* mget)(const matrix *M, size_t i, size_t j);
	OK_CHECK_MATRIX(A);

	mget = (A->order == CblasRowMajor) ? __matrix_get_rowmajor<ok_float> :
		__matrix_get_colmajor<ok_float>;

	for (i = 0; i < A->size1; ++i) {
		for (j = 0; j < A->size2; ++j)
			printf("%e ", mget(A, i, j));
		printf("\n");
	}
	printf("\n");
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
