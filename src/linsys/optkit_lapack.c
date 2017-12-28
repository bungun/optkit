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

ok_status lapack_solve_LU_flagged(void *hdl, matrix *A, vector *x,
	int_vector *pivot, int silence_lapack_err)
{
	lapack_int *ipiv = OK_NULL;
	lapack_int err = 0;
	lapack_int n, nrhs, lda, ldb;

	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(pivot);

	if (A->size1 != A->size2 || x->size != A->size2 || pivot->size != A->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	// err = LAPACKE(gesv)((int) A->order, (lapack_int) A->size1,
	// (lapack_int) 1, A->data, (int) A->ld, pivot->data, x->data, (int) x->stride);

	if (A->order != CblasColMajor)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

	n = (lapack_int) A->size1;
	nrhs = (lapack_int) 1;
	lda = (lapack_int) A->ld;
	ldb = (lapack_int) x->size;
	ipiv = (lapack_int *) pivot->data;

	LAPACK(gesv)(&n, &nrhs, A->data, &lda, ipiv, x->data, &ldb, &err);

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

ok_status lapack_solve_LU(void *hdl, matrix *A, vector *x, int_vector *pivot)
{
	return lapack_solve_LU_flagged(hdl, A, x, pivot, 0);
}

// ok_status lapack_solve_LU_matrix(void *hdl, matrix *A, matrix *X,
//	int_vector *pivot)
// {
// 	lapack_int *pivot = OK_NULL;
// 	lapack_int err = 0;

// 	OK_CHECK_MATRIX(A);
// 	OK_CHECK_MATRIX(X);

// 	if (A->size1 != A->size2 || X->size1 != A->size2)
// 		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
// 	if (A->order != X->order)
// 		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

// 	err = LAPACKE(gesv)((int) A->order, (lapack_int) A->size1,
// 	(lapack_int) X->size2, A->data, (int) A->ld, (lapack_int *) pivot,
// 	X->data, (int) X->ld);

// 	if (err) {
// 		printf("%s%i\n", "LAPACK ERROR: ", (int) err);
// 		return OK_SCAN_ERR( OPTKIT_ERROR_LAPACK );
// 	}
// 	return OPTKIT_SUCCESS;
// }

#ifdef __cplusplus
}
#endif
