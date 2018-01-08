#include "optkit_matrix.h"

template<typename T>
ok_status matrix_submatrix_(matrix_<T> *A_sub, matrix_<T> *A,
	size_t i, size_t j, size_t n1, size_t n2)
{
	if (!A_sub || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (A_sub->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

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
	if (row->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

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
	if (col->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

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
	if (diag->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

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
	if (v->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	v->size = A->size1 * A->size2;
	v->stride = 1;
	v->data = A->data;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_view_vector_(matrix_<T> *A, vector_<T> *v, enum CBLAS_ORDER ord)
{
	if (!A || !v || !v->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	A->size1 = (ord == CblasColMajor) ? v->size : (size_t) 1;
	A->size2 = (ord == CblasColMajor) ? (size_t) 1 : v->size;
	A->ld = v->size;
	A->data = v->data;
	A->order = ord;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_view_array_(matrix_<T> *A, const T *base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord)
{
	if (!A || !base)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	A->size1 = n1;
	A->size2 = n2;
	A->data = (T *) base;
	A->ld = (ord == CblasRowMajor) ? n2 : n1;
	A->order = ord;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_scale_(matrix_<T> *A, T x)
{
	size_t i;
	vector_<T> row_col;
	OK_CHECK_MATRIX(A);
	if (A->order == CblasRowMajor)
		for (i = 0; i < A->size1; ++i) {
			row_col.data = OK_NULL;
			matrix_row_<T>(&row_col, A, i);
			vector_scale_<T>(&row_col, x);
		}
	else
		for (i = 0; i < A->size2; ++i) {
			row_col.data = OK_NULL;
			matrix_column_<T>(&row_col, A, i);
			vector_scale_<T>(&row_col, x);
		}
	return OPTKIT_SUCCESS;
}

template ok_status matrix_row_(vector_<ok_float> *, matrix_<ok_float> *, size_t);
template ok_status matrix_column_(vector_<ok_float> *, matrix_<ok_float> *, size_t);

#ifdef __cplusplus
extern "C" {
#endif

ok_status matrix_alloc(matrix *A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ return matrix_alloc_<ok_float>(A, m, n, ord); }

ok_status matrix_calloc(matrix *A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ return matrix_calloc_<ok_float>(A, m, n, ord); }

ok_status matrix_free(matrix *A)
	{ return matrix_free_<ok_float>(A); }

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

ok_status matrix_view_vector(matrix *A, vector *v, enum CBLAS_ORDER ord)
	{ return matrix_view_vector_<ok_float>(A, v, ord); }

ok_status matrix_view_array(matrix *A, const ok_float *base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord)
	{ return matrix_view_array_<ok_float>(A, base, n1, n2, ord); }

ok_status matrix_scale(matrix *A, ok_float x)
	{ return matrix_scale_<ok_float>(A, x); }

ok_status matrix_scale_left(matrix *A, const vector *v)
	{ return matrix_scale_left_<ok_float>(A, v); }

ok_status matrix_scale_right(matrix *A, const vector *v)
	{ return matrix_scale_right_<ok_float>(A, v); }

ok_status matrix_abs(matrix *A)
{
	vector entries;
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_MATRIX(A);

	entries.data = OK_NULL;
	OK_CHECK_ERR(err, matrix_cast_vector(&entries, A));
	OK_CHECK_ERR(err, vector_abs(&entries));
	return err;
}

ok_status matrix_pow(matrix *A, const ok_float x)
{
	vector entries;
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_MATRIX(A);

	entries.data = OK_NULL;
	OK_CHECK_ERR(err, matrix_cast_vector(&entries, A));
	OK_CHECK_ERR(err, vector_pow(&entries, x));
	return err;
}

#ifdef __cplusplus
}
#endif
