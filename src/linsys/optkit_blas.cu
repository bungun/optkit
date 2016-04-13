#include "optkit_defs_gpu.h"
#include "optkit_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status blas_make_handle(void ** handle)
{
	cublasStatus_t status;
	cublasHandle_t * hdl;
	hdl = (cublasHandle_t *) malloc(sizeof(cublasHandle_t));
	status = cublasCreate(hdl);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization failed\n");
		ok_free(hdl);
		*handle = OK_NULL;
		return OPTKIT_ERROR_CUBLAS;
	} else {
		*handle = (void *) hdl;
		return OPTKIT_SUCCESS;
	}
}

ok_status blas_destroy_handle(void * handle)
{
	cublasDestroy(*(cublasHandle_t *) handle);
	CUDA_CHECK_ERR;
	ok_free(handle);
	return OPTKIT_SUCCESS;
}

/* BLAS LEVEL 1 */
void blas_axpy(void * linalg_handle, ok_float alpha, const vector *x, vector *y)
{
	if (!linalg_handle)
		return;
	CUBLAS(axpy)(*(cublasHandle_t *) linalg_handle, (int) x->size, &alpha,
		x->data, (int) x->stride, y->data, (int) y->stride);
	CUDA_CHECK_ERR;
}

ok_float blas_nrm2(void * linalg_handle, const vector *x)
{
	ok_float result = kZero;
	if (!linalg_handle)
		return OK_NAN;
	CUBLAS(nrm2)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
		(int) x->stride, &result);
	CUDA_CHECK_ERR;
	return result;
}

void blas_scal(void * linalg_handle, const ok_float alpha, vector *x)
{
	if (!linalg_handle)
		return;
	CUBLAS(scal)(*(cublasHandle_t *) linalg_handle, (int) x->size, &alpha,
		x->data, (int) x->stride);
	CUDA_CHECK_ERR;
}

ok_float blas_asum(void * linalg_handle, const vector * x)
{
	ok_float result = kZero;
	if (!linalg_handle)
		return OK_NAN;
	CUBLAS(asum)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
		(int) x->stride, &result);
	CUDA_CHECK_ERR;
	return result;
}

ok_float blas_dot(void * linalg_handle, const vector * x, const vector * y)
{
	ok_float result = kZero;
	if (!linalg_handle)
		return OK_NAN;
	CUBLAS(dot)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
		(int) x->stride, y->data, (int) y->stride, &result);
	CUDA_CHECK_ERR;
	return result;
}

void blas_dot_inplace(void * linalg_handle, const vector * x, const vector * y,
	ok_float * deviceptr_result)
{
	CUBLAS(dot)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
		(int) x->stride, y->data, (int) y->stride, deviceptr_result);
	CUDA_CHECK_ERR;
}

/* BLAS LEVEL 2 */

void blas_gemv(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, const matrix *A, const vector *x, ok_float beta,
	vector *y)
{
	cublasOperation_t tA;
	int s1, s2;

	if (A->order == CblasColMajor)
		tA = (transA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	else
		tA = (transA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

	s1 = (A->order == CblasRowMajor) ? (int) A->size2 : (int) A->size1;
	s2 = (A->order == CblasRowMajor) ? (int) A->size1 : (int) A->size2;

	if (!linalg_handle)
		return;

	CUBLAS(gemv)(*(cublasHandle_t *) linalg_handle, tA, s1, s2, &alpha,
		A->data, (int) A->ld, x->data, (int) x->stride, &beta, y->data,
		(int) y->stride);
	CUDA_CHECK_ERR;
}

void blas_trsv(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, const matrix *A,
	vector *x)
{
	cublasOperation_t tA;
	cublasDiagType_t di;
	cublasFillMode_t ul;

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

	if (!linalg_handle)
		return;

	CUBLAS(trsv)(*(cublasHandle_t *) linalg_handle, ul, tA, di,
		(int) A->size1, A->data, (int) A->ld, x->data, (int) x->stride);
	CUDA_CHECK_ERR;
}

void blas_sbmv(void * linalg_handle, enum CBLAS_ORDER order,
	enum CBLAS_UPLO uplo, const size_t num_superdiag, const ok_float alpha,
	const vector * vecA, const vector * x, const ok_float beta, vector * y)
{
	cublasFillMode_t ul;
	if (order == CblasRowMajor)
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
	else
		ul = (uplo == CblasLower) ?
		     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

	CUBLAS(sbmv)(*(cublasHandle_t *) linalg_handle, ul,
	       (int) y->size, (int) num_superdiag, &alpha,
	       vecA->data, (int) (num_superdiag + 1),
	       x->data, (int) x->stride, &beta, y->data, (int) y->stride);
}

void blas_diagmv(void * linalg_handle, const ok_float alpha,
	const vector * vecA, const vector * x, const ok_float beta, vector * y)
{
	blas_sbmv(linalg_handle, CblasColMajor, CblasLower, 0, alpha, vecA, x,
		beta, y);
}

/* BLAS LEVEL 3 */
void blas_syrk(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, ok_float alpha, const matrix * A,
	ok_float beta, matrix * C)
{

	cublasOperation_t tA;
	cublasFillMode_t ul;
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


	if (!linalg_handle)
		return;

	if ( !__matrix_order_compat(A, C, "A", "C", "blas_syrk") )
		return;


	CUBLAS(syrk)(*(cublasHandle_t *) linalg_handle, ul, tA, (int) C->size2,
		k, &alpha, A->data, (int) A->ld, &beta, C->data, (int) C->ld);

	CUDA_CHECK_ERR;
}

void blas_gemm(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	enum CBLAS_TRANSPOSE transB, ok_float alpha, const matrix * A,
	const matrix * B, ok_float beta, matrix * C)
{
	cublasOperation_t tA, tB;
	int s1, s2;

	const int k = (transA == CblasNoTrans) ?
		      (int) A->size2 : (int) A->size1;

	s1 = (A->order == CblasRowMajor) ? (int) C->size2 : (int) C->size1;
	s2 = (A->order == CblasRowMajor) ? (int) C->size1 : (int) C->size2;
	if (A->order == CblasColMajor) {
		tA = transA == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
		tB = transB == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
	} else {
		tA = transB == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
		tB = transA == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
	}

	if (!linalg_handle)
		return;

	if (!__matrix_order_compat(A, B, "A", "B", "blas_gemm") ||
		!__matrix_order_compat(A, C, "A", "C", "blas_gemm"))
		return;

	CUBLAS(gemm)(*(cublasHandle_t *) linalg_handle, tA, tB, s1, s2, k,
		&alpha, A->data, (int) A->ld, B->data, (int) B->ld, &beta,
		C->data, (int) C->ld);

	CUDA_CHECK_ERR;
}

void blas_trsm(void * linalg_handle, enum CBLAS_SIDE Side, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, ok_float alpha,
	const matrix *A, matrix *B)
{
	printf("Method `blas_trsm()` not implemented for GPU\n");
}


#ifdef __cplusplus
}
#endif