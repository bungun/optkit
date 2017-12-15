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
ok_status matrix_submatrix_(matrix_<T> *A_sub, matrix_<T> *A,
    size_t i, size_t j, size_t n1, size_t n2)
{
	if (!A_sub || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	A_sub->size1 = n1;
	A_sub->size2 = n2;
	A_sub->ld = A->ld;
	A_sub->data = (A->order == CblasRowMajor) ?
		      A->data + (i * A->ld) + j :
		      A->data + i + (j * A->ld);
	A_sub->order = A->order;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_row_(vector_<T> *row, matrix_<T> *A, size_t i)
{
	if (!row || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	row->size = A->size2;
	row->stride = (A->order == CblasRowMajor) ? 1 : A->ld;
	row->data = (A->order == CblasRowMajor) ?
		    A->data + (i * A->ld) :
		    A->data + i;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_column_(vector_<T> *col, matrix_<T> *A, size_t j)
{
	if (!col || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	col->size = A->size1;
	col->stride = (A->order == CblasRowMajor) ? A->ld : 1;
	col->data = (A->order == CblasRowMajor) ?
		    A->data + j :
		    A->data + (j * A->ld);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_diagonal_(vector_<T> *diag, matrix_<T> *A)
{
	if (!diag || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	diag->data = A->data;
	diag->stride = A->ld + 1;
	diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
	return OPTKIT_SUCCESS;

}

template<typename T>
ok_status matrix_cast_vector_(vector_<T> *v, matrix_<T> *A)
{
	if (!v || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	v->size = A->size1 * A->size2;
	v->stride = 1;
	v->data = A->data;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_view_array_(matrix_<T> *A, const T *base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord)
{
	if (!A || !base)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	A->size1 = n1;
	A->size2 = n2;
	A->data = (T *) base;
	A->ld = (ord == CblasRowMajor) ? n2 : n1;
	A->order = ord;
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
ok_status matrix_scale_(matrix_<T> *A, T x)
{
	size_t i;
	vector_<T> row_col;
	row_col.data = OK_NULL;
	OK_CHECK_MATRIX(A);

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
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_scale_left_(matrix_<T> *A, const vector_<T> *v)
{
	size_t i;
	vector_<T> row;
	row.data = OK_NULL;

	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);
	if (A->size1 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	for(i = 0; i < A->size1; ++i) {
		matrix_row_<T>(&row, A, i);
		vector_scale_<T>(&row, v->data[i]);
	}
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_scale_right_(matrix_<T> *A, const vector_<T> *v)
{
	size_t i;
	vector_<T> col;
	col.data = OK_NULL;

	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);
	if (A->size2 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	for(i = 0; i < A->size2; ++i) {
		matrix_column_<T>(&col, A, i);
		vector_scale_<T>(&col, v->data[i]);
	}
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

ok_status matrix_alloc(matrix *A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ return matrix_alloc_<ok_float>(A, m, n, ord); }

ok_status matrix_calloc(matrix *A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ return matrix_calloc_<ok_float>(A, m, n, ord); }

ok_status matrix_free(matrix *A)
	{ return matrix_free_<ok_float>(A); }

ok_status matrix_submatrix(matrix *A_sub, matrix *A, size_t i, size_t j,
	size_t n1, size_t n2)
	{ return matrix_submatrix_<ok_float>(A_sub, A, i, j, n1, n2); }

ok_status matrix_row(vector *row, matrix *A, size_t i)
	{ return matrix_row_<ok_float>(row, A, i); }

ok_status matrix_column(vector *col, matrix *A, size_t j)
	{ return matrix_column_<ok_float>(col, A, j); }

ok_status matrix_diagonal(vector *diag, matrix *A)
	{ return matrix_diagonal_<ok_float>(diag, A); }

ok_status matrix_cast_vector(vector *v, matrix *A)
	{ return matrix_cast_vector_<ok_float>(v, A); }

ok_status matrix_view_array(matrix *A, const ok_float *base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord)
	{ return matrix_view_array_<ok_float>(A, base, n1, n2, ord); }

ok_status matrix_set_all(matrix *A, ok_float x)
	{ return matrix_set_all_<ok_float>(A, x); }

ok_status matrix_memcpy_mm(matrix *A, const matrix *B)
	{ return matrix_memcpy_mm_<ok_float>(A, B); }

ok_status matrix_memcpy_ma(matrix *A, const ok_float *B,
	const enum CBLAS_ORDER ord)
	{ return matrix_memcpy_ma_<ok_float>(A, B, ord); }

ok_status matrix_memcpy_am(ok_float *A, const matrix *B,
	const enum CBLAS_ORDER ord)
	{ return matrix_memcpy_am_<ok_float>(A, B, ord); }

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

ok_status matrix_scale(matrix *A, ok_float x)
	{ return matrix_scale_<ok_float>(A, x); }

ok_status matrix_scale_left(matrix *A, const vector *v)
	{ return matrix_scale_left_<ok_float>(A, v); }

ok_status matrix_scale_right(matrix *A, const vector *v)
	{ return matrix_scale_right_<ok_float>(A, v); }

ok_status matrix_abs(matrix *A)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t i;
	vector row_col;
	row_col.data = OK_NULL;

	OK_CHECK_MATRIX(A);

	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1 && !err; ++i) {
			matrix_row(&row_col, A, i);
			vector_abs(&row_col);
		}
	else
		for(i = 0; i < A->size2 && !err; ++i) {
			matrix_column(&row_col, A, i);
			vector_abs(&row_col);
		}
	return OPTKIT_SUCCESS;
}

ok_status matrix_pow(matrix *A, const ok_float x)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t i;
	vector row_col;
	row_col.data = OK_NULL;

	OK_CHECK_MATRIX(A);

	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1 && !err; ++i) {
			matrix_row(&row_col, A, i);
			vector_pow(&row_col, x);
	}
	else
		for(i = 0; i < A->size2 && !err; ++i) {
			matrix_column(&row_col, A, i);
			vector_pow(&row_col, x);
	}
	return OK_SCAN_ERR( err );
}

#ifdef __cplusplus
}
#endif
