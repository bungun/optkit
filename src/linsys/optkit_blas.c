#include "optkit_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status blas_make_handle(void **linalg_handle)
{
	*linalg_handle = OK_NULL;
	return OPTKIT_SUCCESS;
}

ok_status blas_destroy_handle(void *linalg_handle)
{
	linalg_handle = OK_NULL;
	return OPTKIT_SUCCESS;
}

/* BLAS LEVEL 1 */

ok_status blas_axpy(void *linalg_handle, ok_float alpha, const vector *x,
	vector *y)
{
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if (x->size != y->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	CBLAS(axpy)((int) x->size, alpha, x->data, (int) x->stride, y->data,
		(int) y->stride);
	return OPTKIT_SUCCESS;
}

ok_status blas_nrm2(void *linalg_handle, const vector *x, ok_float *result)
{
	OK_CHECK_VECTOR(x);
	*result = CBLAS(nrm2)((int) x->size, x->data, (int) x->stride);
	return OPTKIT_SUCCESS;
}

ok_status blas_scal(void *linalg_handle, const ok_float alpha, vector *x)
{
	OK_CHECK_VECTOR(x);
	CBLAS(scal)((int) x->size, alpha, x->data, (int) x->stride);
	return OPTKIT_SUCCESS;
}

ok_status blas_asum(void *linalg_handle, const vector *x, ok_float *result)
{
	OK_CHECK_VECTOR(x);
	*result = CBLAS(asum)((int) x->size, x->data, (int) x->stride);
	return OPTKIT_SUCCESS;
}

ok_status blas_dot(void *linalg_handle, const vector *x, const vector *y,
	ok_float *result)
{
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if (x->size != y->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	*result = CBLAS(dot)((int) x->size, x->data, (int) x->stride, y->data,
		(int) y->stride);
	return OPTKIT_SUCCESS;
}

// ok_status blas_dot_inplace(void *linalg_handle, const vector *x,
// 	const vector *y, ok_float *deviceptr_result)
// {
// 	OK_CHECK_VECTOR(x);
// 	OK_CHECK_VECTOR(y);
// 	if (x->size != y->size)
// 		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

// 	*deviceptr_result = CBLAS(dot)((int) x->size, x->data, (int) x->stride,
// 		y->data, (int) y->stride);

// 	return OPTKIT_SUCCESS;
// }

/* BLAS LEVEL 2 */

ok_status blas_gemv(void *linalg_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, const matrix *A, const vector *x, ok_float beta,
	vector *y)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if ((transA == CblasNoTrans &&
		(A->size1 != y->size || A->size2 != x->size)) ||
	    (transA == CblasTrans &&
		(A->size2 != y->size || A->size1 != x->size)))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	CBLAS(gemv)(A->order, transA, (int) A->size1, (int) A->size2, alpha,
		A->data, (int) A->ld, x->data, (int) x->stride, beta,
		y->data, (int) y->stride);
	return OPTKIT_SUCCESS;
}

ok_status blas_trsv(void *linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, const matrix *A,
	vector *x)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(x);
	if (A->size1 != A->size2 || A->size1 != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	CBLAS(trsv)(A->order, uplo, transA, Diag, (int) A->size1, A->data,
		(int) A->ld, x->data, (int) x->stride);
	return OPTKIT_SUCCESS;
}

ok_status blas_sbmv(void *linalg_handle, enum CBLAS_ORDER order,
	enum CBLAS_UPLO uplo, const size_t num_superdiag, const ok_float alpha,
	const vector *vecA, const vector *x, const ok_float beta, vector *y)
{
	size_t lenA;
	OK_CHECK_VECTOR(vecA);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);

	/*
	 * require:
	 *	- x.size == y.size
	 *	- num_superdiag == 0 and vecA.size >= y.size OR
	 *	- num_superdiag > 0 and vecA.size >= \sum_i=1^k y.size - i
	 */
	lenA = y->size * (num_superdiag + 1);
	if (num_superdiag > 0 && num_superdiag < y->size)
		lenA -= ((num_superdiag) * (num_superdiag + 1)) / 2;

	if (x->size != y->size || vecA->size < lenA)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	CBLAS(sbmv)(order, uplo, (int) y->size, (int) num_superdiag, alpha,
		vecA->data, (int) num_superdiag + 1, x->data, (int) x->stride,
		beta, y->data, (int) y->stride);
	return OPTKIT_SUCCESS;
}

ok_status blas_diagmv(void *linalg_handle, const ok_float alpha,
	const vector *vecA, const vector *x, const ok_float beta, vector *y)
{
	OK_CHECK_VECTOR(vecA);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if (vecA->size != y->size || x->size != y->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	return blas_sbmv(linalg_handle, CblasColMajor, CblasLower, 0, alpha,
		vecA, x, beta, y);
}

/* BLAS LEVEL 3 */

ok_status blas_syrk(void *linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, ok_float alpha, const matrix *A,
	ok_float beta, matrix *C)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	const int k = (transA == CblasNoTrans) ? (int) A->size2:(int) A->size1;
	if (A->order != C->order)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );
	if (C->size1 != C->size2 ||
		(transA == CblasNoTrans && A->size1 != C->size1) ||
		(transA == CblasTrans && A->size2 != C->size2))
		return OK_SCAN_ERR ( OPTKIT_ERROR_DIMENSION_MISMATCH );

	CBLAS(syrk)(A->order, uplo, transA, (int) C->size2, k, alpha, A->data,
		(int) A->ld, beta, C->data, (int) C->ld);
	return OPTKIT_SUCCESS;
}

ok_status blas_gemm(void *linalg_handle, enum CBLAS_TRANSPOSE transA,
	enum CBLAS_TRANSPOSE transB, ok_float alpha, const matrix *A,
	const matrix *B, ok_float beta, matrix *C)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(B);
	OK_CHECK_MATRIX(C);
	const int inner_dim = (transA == CblasNoTrans) ?
		(int) A->size2 : (int) A->size1;

	if (A->order != C->order || B->order != C->order)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

	if (transA == CblasNoTrans && transB == CblasNoTrans)
		if (A->size1 != C->size1 || A->size2 != B->size1 ||
			B->size2 != C->size2)
			return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	if (transA == CblasNoTrans && transB == CblasTrans)
		if (A->size1 != C->size1 || A->size2 != B->size2 ||
			B->size1 != C->size2)
			return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	if (transA == CblasTrans && transB == CblasNoTrans)
		if (A->size2 != C->size1 || A->size1 != B->size1 ||
			B->size2 != C->size2)
			return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	if (transA == CblasTrans && transB == CblasTrans)
		if (A->size2 != C->size1 || A->size1 != B->size2 ||
			B->size1 != C->size2)
			return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	CBLAS(gemm)(A->order, transA, transB, (int) C->size1,
		(int) C->size2, inner_dim, alpha, A->data, (int) A->ld,
		B->data, (int) B->ld, beta, C->data, (int) C->ld);
	return OPTKIT_SUCCESS;
}

ok_status blas_trsm(void *linalg_handle, enum CBLAS_SIDE side,
	enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG diag,
	ok_float alpha, const matrix *A, matrix *B)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(B);
	if (A->order != B->order)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );
	if ((side == CblasLeft && A->size1 != B->size1) ||
		(side == CblasRight && A->size1 != B->size2))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	CBLAS(trsm)(A->order, side, uplo, transA, diag, (int) B->size1,
		(int) B->size2, alpha, A->data,(int) A->ld, B->data,
		(int) B->ld);
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
