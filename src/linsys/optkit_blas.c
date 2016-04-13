#include "optkit_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

static int __matrix_order_compat(const matrix * A, const matrix * B,
	const char * nm_A, const char * nm_B, const char * nm_routine)
{
	if (A->order == B->order)
		return 1;

	printf("OPTKIT ERROR (%s) matrices %s and %s must have same layout.\n",
		nm_routine, nm_A, nm_B);
	return 0;
}

ok_status blas_make_handle(void ** linalg_handle)
{
	*linalg_handle = OK_NULL;
	return OPTKIT_SUCCESS;
}

ok_status blas_destroy_handle(void * linalg_handle)
{
	linalg_handle = OK_NULL;
	return OPTKIT_SUCCESS;
}

/* BLAS LEVEL 1 */

void blas_axpy(void * linalg_handle, ok_float alpha, const vector * x, vector *y)
{
	CBLAS(axpy)((int) x->size, alpha, x->data, (int) x->stride, y->data,
		(int) y->stride);
}

ok_float blas_nrm2(void * linalg_handle, const vector * x)
{
	return CBLAS(nrm2)((int) x->size, x->data, (int) x->stride);
}

void blas_scal(void * linalg_handle, const ok_float alpha, vector * x)
{
	CBLAS(scal)((int) x->size, alpha, x->data, (int) x->stride);
}

ok_float blas_asum(void * linalg_handle, const vector * x)
{
	return CBLAS(asum)((int) x->size, x->data, (int) x->stride);
}

ok_float blas_dot(void * linalg_handle, const vector * x, const vector * y)
{
	return CBLAS(dot)((int) x->size, x->data, (int) x->stride, y->data,
		(int) y->stride);
}

void blas_dot_inplace(void * linalg_handle, const vector * x, const vector * y,
	ok_float * deviceptr_result)
{
	*deviceptr_result = CBLAS(dot)((int) x->size, x->data, (int) x->stride,
		y->data, (int) y->stride);
}

/* BLAS LEVEL 2 */

void blas_gemv(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, const matrix * A, const vector * x, ok_float beta,
	vector * y)
{
	CBLAS(gemv)(A->order, transA, (int) A->size1, (int) A->size2, alpha,
		A->data, (int) A->ld, x->data, (int) x->stride, beta,
		y->data, (int) y->stride);
}

void blas_trsv(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, const matrix * A,
	vector *x)
{
	CBLAS(trsv)(A->order, uplo, transA, Diag, (int) A->size1, A->data,
		(int) A->ld, x->data, (int) x->stride);
}

void blas_sbmv(void * linalg_handle, enum CBLAS_ORDER order,
	enum CBLAS_UPLO uplo, const size_t num_superdiag, const ok_float alpha,
	const vector * vecA, const vector * x, const ok_float beta,
	vector * y)
{
	CBLAS(sbmv)(order, uplo, (int) y->size, (int) num_superdiag, alpha,
		vecA->data, (int) num_superdiag + 1, x->data, (int) x->stride,
		beta, y->data, (int) y->stride);
}

void blas_diagmv(void * linalg_handle, const ok_float alpha,
	const vector * vecA, const vector * x, const ok_float beta,
	vector * y)
{
	blas_sbmv(linalg_handle, CblasColMajor, CblasLower, 0, alpha, vecA, x,
		beta, y);
}

/* BLAS LEVEL 3 */

void blas_syrk(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, ok_float alpha, const matrix * A,
	ok_float beta, matrix * C)
{
	const int k = (transA == CblasNoTrans) ? (int) A->size2 :
					 (int) A->size1;

	if (!( __matrix_order_compat(A, C, "A", "C", "blas_syrk") ))
		return;

	CBLAS(syrk)(A->order, uplo, transA, (int) C->size2, k, alpha, A->data,
		(int) A->ld, beta, C->data, (int) C->ld);
}

void blas_gemm(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	enum CBLAS_TRANSPOSE transB, ok_float alpha, const matrix * A,
	const matrix * B, ok_float beta, matrix * C)
{
	const int NA = (transA == CblasNoTrans) ? (int) A->size2 :
						  (int) A->size1;

	if (!( __matrix_order_compat(A, B, "A", "B", "gemm") &&
		__matrix_order_compat(A, C, "A", "C", "blas_gemm") ))
		return;
	CBLAS(gemm)(A->order, transA, transB, (int) C->size1, (int) C->size2,
		NA, alpha, A->data, (int) A->ld, B->data, (int) B->ld,
		beta, C->data, (int) C->ld);
}

void blas_trsm(void * linalg_handle, enum CBLAS_SIDE Side,
	enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag,
	ok_float alpha, const matrix *A, matrix *B)
{
	if (!( __matrix_order_compat(A, B, "A", "B", "blas_trsm") ))
		return;
	CBLAS(trsm)(A->order, Side, uplo, transA, Diag, (int) B->size1,
		(int) B->size2, alpha, A->data,(int) A->ld, B->data,
		(int) B->ld);
}

#ifdef __cplusplus
}
#endif
