#ifndef OPTKIT_LINSYS_BLAS_H_
#define OPTKIT_LINSYS_BLAS_H_

#include "optkit_vector.h"
#include "optkit_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* BLAS context */
ok_status blas_make_handle(void ** linalg_handle);
ok_status blas_destroy_handle(void * linalg_handle);

/* BLAS LEVEL 1 */
void blas_axpy(void * linalg_handle, ok_float alpha, const vector * x,
	vector * y);
ok_float blas_nrm2(void * linalg_handle, const vector *x);
void blas_scal(void * linalg_handle, const ok_float alpha, vector * x);
ok_float blas_asum(void * linalg_handle, const vector * x);
ok_float blas_dot(void * linalg_handle, const vector * x, const vector * y);
void blas_dot_inplace(void * linalg_handle, const vector * x, const vector * y,
	ok_float * deviceptr_result);

/* BLAS LEVEL 2 */
void blas_gemv(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, const matrix * A, const vector * x, ok_float beta,
	vector * y);

void blas_trsv(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, const matrix * A,
	vector * x);

void blas_sbmv(void * linalg_handle, enum CBLAS_ORDER order,
	enum CBLAS_UPLO uplo, const size_t num_superdiag, const ok_float alpha,
	const vector * vecA, const vector * x, const ok_float beta, vector * y);

void blas_diagmv(void * linalg_handle, const ok_float alpha,
	const vector * vecA, const vector * x, const ok_float beta, vector * y);

/* BLAS LEVEL 3 */
void blas_syrk(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, ok_float alpha, const matrix *A,
	ok_float beta, matrix *C);

void blas_gemm(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	enum CBLAS_TRANSPOSE transB, ok_float alpha, const matrix *A,
	const matrix *B, ok_float beta, matrix *C);

void blas_trsm(void * linalg_handle, enum CBLAS_SIDE Side, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, ok_float alpha,
	const matrix *A, matrix *B);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_LINSYS_BLAS_H_ */
