#include "optkit_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status lapack_solve_LU_flagged(void *hdl, matrix *A, vector *x,
	int_vector *pivot, int silence_lapack_err)
{
	ok_status err = OPTKIT_SUCCESS;
	matrix X;

	X.data = OK_NULL;
	OK_CHECK_ERR(err, matrix_view_vector(&X, x, CblasColMajor));
	OK_CHECK_ERR(err, lapack_solve_LU_matrix_flagged(hdl, A, &X, pivot,
		silence_lapack_err));
	return err;
}

ok_status lapack_solve_LU_matrix(void *hdl, matrix *A, matrix *X, int_vector *pivot)
{
	return lapack_solve_LU_matrix_flagged(hdl, A, X, pivot, 0);
}

ok_status lapack_solve_LU(void *hdl, matrix *A, vector *x, int_vector *pivot)
{
	return lapack_solve_LU_flagged(hdl, A, x, pivot, 0);
}

ok_status lapack_cholesky_decomp(void *hdl, matrix *A)
{
	return lapack_cholesky_decomp_flagged(hdl, A, 0);
}



#ifdef __cplusplus
}
#endif
