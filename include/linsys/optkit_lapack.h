#ifndef OPTKIT_LINSYS_LAPACK_H_
#define OPTKIT_LINSYS_LAPACK_H_

#include "optkit_vector.h"
#include "optkit_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

void LAPACK(gesv)(lapack_int *n, lapack_int *nrhs, ok_float *a, lapack_int *lda,
	lapack_int *ipiv, ok_float *b, lapack_int *ldb, lapack_int *info);

/* LAPACK context */
ok_status lapack_make_handle(void **lapack_handle);
ok_status lapack_destroy_handle(void *lapack_handle);

/* QR SOLVE */

/* LU SOLVE */
ok_status lapack_solve_LU_flagged(void *hdl, matrix *A, vector *x,
	int_vector *pivot, int silence_lapack_err);
ok_status lapack_solve_LU(void *hdl, matrix *A, vector *x, int_vector *pivot);
ok_status lapack_solve_LU_matrix(void *hdl, matrix *A, matrix *X,
	int_vector *pivot);

/* CHOLESKY SOLVE */

/* EVD */


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_LINSYS_LAPACK_H_ */
