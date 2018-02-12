#include "optkit_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status lapack_make_handle(void **lapack_handle)
{
	return OPTKIT_SUCCESS;
}

ok_status lapack_destroy_handle(void *lapack_handle)
{
	return OPTKIT_SUCCESS;
}

/*
 * TODO: no support for row-major matrices when compiling against fortran
 * LAPACK, as opposed to C LAPACKE interface which accomodates array layout
 */
ok_status lapack_solve_LU_matrix_flagged(void *hdl, matrix *A, matrix *X,
	int_vector *pivot, int silence_lapack_err)
{
	lapack_int *ipiv = OK_NULL;
	lapack_int err = 0;
	lapack_int n, nrhs, lda, ldb;

	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(X);
	OK_CHECK_VECTOR(pivot);

	if (A->size1 != A->size2 || X->size1 != A->size2 || pivot->size != A->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

#ifndef OK_C_LAPACKE
	if (A->order != CblasColMajor)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );
#endif /* OK_C_LAPACKE */

	n = (lapack_int) A->size1;
	nrhs = (lapack_int) X->size2;
	lda = (lapack_int) A->ld;
	ldb = (lapack_int) X->ld;
	ipiv = (lapack_int *) pivot->data;

#ifdef OK_C_LAPACKE
	err = LAPACKE(gesv)((int) A->order, n, nrhs, A->data, lda, ipiv,
		X->data, ldb);
#else
	LAPACK(gesv)(&n, &nrhs, A->data, &lda, ipiv, X->data, &ldb, &err);
#endif /* OK_C_LAPACKE */

	if (err) {
		if (silence_lapack_err) {
			return OPTKIT_ERROR_LAPACK;
		} else {
			printf("%s%i\n", "LAPACK ERROR: ", (int) err);
			return OK_SCAN_ERR( OPTKIT_ERROR_LAPACK );
		}
	}
	return OPTKIT_SUCCESS;
}

ok_status lapack_cholesky_decomp_flagged(void *hdl, matrix *A,
	int silence_lapack_err)
{
	lapack_int n;
	lapack_int err = 0;
	char uplo;
	OK_CHECK_MATRIX(A);
	if (A->size1 != A->size2)
		return OK_SCAN_ERR(OPTKIT_ERROR_DIMENSION_MISMATCH);

	uplo = (A->order == CblasColMajor) ? 'L' : 'U';
	n = (lapack_int) A->size1;


#ifdef OK_C_LAPACKE
	err = LAPACKE(potrf)((int) CblasColMajor, uplo, n, A->data, n);
#else
	LAPACK(potrf)(&uplo, &n, A->data, &n, &err);
#endif /* OK_C_LAPACKE */

	if (err && silence_lapack_err) {
		return OPTKIT_ERROR_LAPACK;
	} else if (err) {
		printf("%s%i\n", "LAPACK ERROR: ", (int) err);
		return OK_SCAN_ERR(OPTKIT_ERROR_LAPACK);
	}
	return OPTKIT_SUCCESS;
}

ok_status lapack_cholesky_svx(void *hdl, const matrix *L, vector *x)
{
	lapack_int n, n_rhs;
	lapack_int err = 0;
	char uplo;
	OK_CHECK_MATRIX(L);
	OK_CHECK_VECTOR(x);
	if (L->size1 != L->size2 || x->size != L->size2)
		return OK_SCAN_ERR(OPTKIT_ERROR_DIMENSION_MISMATCH);

	uplo = (L->order == CblasColMajor) ? 'L' : 'U';
	n = (lapack_int) L->size1;
	n_rhs = 1;

#ifdef OK_C_LAPACKE
	err = LAPACKE(potrs)((int) CblasColMajor, uplo, n, n_rhs, A->data, n,
		x->data, n);
#else
	LAPACK(potrs)(&uplo, &n, &n_rhs, L->data, &n, x->data, &n, &err);
#endif /* OK_C_LAPACKE */

	if (err) {
		printf("%s%i\n", "LAPACK ERROR: ", (int) err);
		return OK_SCAN_ERR(OPTKIT_ERROR_LAPACK);
	}
	return OPTKIT_SUCCESS;
}
