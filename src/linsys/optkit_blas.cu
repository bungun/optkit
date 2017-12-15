#include "optkit_defs_gpu.h"
#include "optkit_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status blas_make_handle(void **handle)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasHandle_t *hdl;
	hdl = (cublasHandle_t *) malloc(sizeof(cublasHandle_t));
	err = OK_SCAN_CUBLAS( cublasCreate(hdl) );
	if (err) {
		printf("CUBLAS initialization failed\n");
		ok_free(hdl);
		*handle = OK_NULL;
		return err;
	} else {
		*handle = (void *) hdl;
		return OPTKIT_SUCCESS;
	}
}

ok_status blas_destroy_handle(void *handle)
{
	OK_CHECK_PTR(handle);
	ok_status err = OK_SCAN_CUBLAS(
		cublasDestroy(*(cublasHandle_t *) handle) );
	ok_free(handle);
	return err;
}

/* BLAS LEVEL 1 */
ok_status blas_axpy(void *linalg_handle, ok_float alpha, const vector *x,
	vector *y)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if (x->size != y->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	err = OK_SCAN_CUBLAS( CUBLAS(axpy)(*(cublasHandle_t *) linalg_handle,
		(int) x->size, &alpha, x->data, (int) x->stride, y->data,
		(int) y->stride) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_nrm2(void *linalg_handle, const vector *x, ok_float *result)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_VECTOR(x);

	err = OK_SCAN_CUBLAS( CUBLAS(nrm2)(*(cublasHandle_t *) linalg_handle,
		(int) x->size, x->data, (int) x->stride, result) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_scal(void *linalg_handle, const ok_float alpha, vector *x)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_VECTOR(x);

	err = OK_SCAN_CUBLAS( CUBLAS(scal)(*(cublasHandle_t *) linalg_handle,
		(int) x->size, &alpha, x->data, (int) x->stride) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_asum(void *linalg_handle, const vector *x, ok_float *result)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_VECTOR(x);

	err = OK_SCAN_CUBLAS( CUBLAS(asum)(*(cublasHandle_t *) linalg_handle,
		(int) x->size, x->data, (int) x->stride, result) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_dot(void *linalg_handle, const vector *x, const vector *y,
	ok_float *result)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);

	err = OK_SCAN_CUBLAS( CUBLAS(dot)(*(cublasHandle_t *) linalg_handle,
		(int) x->size, x->data, (int) x->stride, y->data,
		(int) y->stride, result) );
	cudaDeviceSynchronize();
	return err;
}

// void blas_dot_inplace(void *linalg_handle, const vector *x, const vector *y,
// 	ok_float *deviceptr_result)
// {
// 	CUBLAS(dot)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
// 		(int) x->stride, y->data, (int) y->stride, deviceptr_result);
// 	cudaDeviceSynchronize();
// 	return err;
// }

/* BLAS LEVEL 2 */

ok_status blas_gemv(void *linalg_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, const matrix *A, const vector *x, ok_float beta,
	vector *y)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasOperation_t tA;
	int s1, s2;

	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if ((transA == CblasNoTrans &&
		(A->size1 != y->size || A->size2 != x->size)) ||
	    (transA == CblasTrans &&
		(A->size2 != y->size || A->size1 != x->size)))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if (A->order == CblasColMajor)
		tA = (transA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	else
		tA = (transA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

	s1 = (A->order == CblasRowMajor) ? (int) A->size2 : (int) A->size1;
	s2 = (A->order == CblasRowMajor) ? (int) A->size1 : (int) A->size2;

	err = OK_SCAN_CUBLAS( CUBLAS(gemv)(*(cublasHandle_t *) linalg_handle,
		tA, s1, s2, &alpha, A->data, (int) A->ld, x->data,
		(int) x->stride, &beta, y->data, (int) y->stride) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_trsv(void *linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, const matrix *A,
	vector *x)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasOperation_t tA;
	cublasDiagType_t di;
	cublasFillMode_t ul;

	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(x);
	if (A->size1 != A->size2 || A->size1 != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if (A->order == CblasColMajor) {
		tA = (transA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
	} else {
		tA = (transA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
	}

	di = Diag == CblasNonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;

	err = OK_SCAN_CUBLAS( CUBLAS(trsv)(*(cublasHandle_t *) linalg_handle,
		ul, tA, di, (int) A->size1, A->data, (int) A->ld, x->data,
		(int) x->stride) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_sbmv(void *linalg_handle, enum CBLAS_ORDER order,
	enum CBLAS_UPLO uplo, const size_t num_superdiag, const ok_float alpha,
	const vector *vecA, const vector *x, const ok_float beta, vector *y)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasFillMode_t ul;
	size_t lenA;

	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_MATRIX(vecA);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);

	/*
	 * require:
	 *	- x.size == y.size
	 *	- num_superdiag == 0 and vecA.size == y.size
	 *	- num_superdiag > 0 and vecA.size == \sum_i=1^k y.size - i
	 */
	lenA = y->size * (num_superdiag + 1);
	if (num_superdiag > 0 && num_superdiag < y->size)
		lenA -= ((num_superdiag) * (num_superdiag + 1)) / 2;

	if (x->size != y->size || vecA->size < lenA)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if (order == CblasRowMajor)
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
	else
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

	err = OK_SCAN_CUBLAS( CUBLAS(sbmv)(*(cublasHandle_t *) linalg_handle,
		ul, (int) y->size, (int) num_superdiag, &alpha, vecA->data,
		(int) (num_superdiag + 1), x->data, (int) x->stride, &beta,
		y->data, (int) y->stride) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_diagmv(void *linalg_handle, const ok_float alpha,
	const vector *vecA, const vector *x, const ok_float beta, vector *y)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_VECTOR(vecA);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if (vecA->size != y->size || x->size != y->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	err = OK_SCAN_ERR( blas_sbmv(linalg_handle, CblasColMajor,
		CblasLower, 0, alpha, vecA, x, beta, y) );
	cudaDeviceSynchronize();
	return err;
}

/* BLAS LEVEL 3 */
ok_status blas_syrk(void *linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, ok_float alpha, const matrix *A,
	ok_float beta, matrix *C)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasOperation_t tA;
	cublasFillMode_t ul;

	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(C);
	if (A->order != C->order)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

	const int k = (transA == CblasNoTrans) ?
		      (int) A->size2 : (int) A->size1;

	if (A->order == CblasColMajor) {
		tA = (transA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
	} else {
		tA = (transA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
	}

	err = OK_SCAN_CUBLAS( CUBLAS(syrk)(*(cublasHandle_t *) linalg_handle,
		ul, tA, (int) C->size2, k, &alpha, A->data, (int) A->ld, &beta,
		C->data, (int) C->ld) );
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_gemm(void *linalg_handle, enum CBLAS_TRANSPOSE transA,
	enum CBLAS_TRANSPOSE transB, ok_float alpha, const matrix *A,
	const matrix *B, ok_float beta, matrix *C)
{
	ok_status err = OPTKIT_SUCCESS;
	cublasOperation_t tA = transA == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t tB = transB == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;

	OK_CHECK_PTR(linalg_handle);
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(B);
	OK_CHECK_MATRIX(C);

	const int k = (transA == CblasNoTrans) ?
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

	if (A->order == CblasColMajor) {
		err = OK_SCAN_CUBLAS( CUBLAS(gemm)(
			*(cublasHandle_t *) linalg_handle, tA, tB,
			(int) C->size1, (int) C->size2, k, &alpha, A->data,
			(int) A->ld, B->data, (int) B->ld, &beta, C->data,
			(int) C->ld) );
	} else {
		err = OK_SCAN_CUBLAS( CUBLAS(gemm)(
			*(cublasHandle_t *) linalg_handle, tB, tA,
			(int) C->size2, (int) C->size1, k, &alpha, B->data,
			(int) B->ld, A->data, (int) A->ld, &beta, C->data,
			(int) C->ld) );
	}
	cudaDeviceSynchronize();
	return err;
}

ok_status blas_trsm(void *linalg_handle, enum CBLAS_SIDE Side,
	enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag,
	ok_float alpha, const matrix *A, matrix *B)
{
	printf("\nMethod `blas_trsm()` not implemented for GPU\n");
	return OPTKIT_ERROR;
}

#ifdef __cplusplus
}
#endif
