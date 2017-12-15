#ifndef OPTKIT_LINSYS_DENSE_H_
#define OPTKIT_LINSYS_DENSE_H_

#include "optkit_vector.h"
#include "optkit_matrix.h"
#include "optkit_blas.h"
#include "optkit_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status linalg_cholesky_decomp(void *linalg_handle, matrix *A);
ok_status linalg_cholesky_svx(void *linalg_handle, const matrix *L,
	vector *x);

/* TODO: consider changing this to matrix_reduce_unary(const enum t, matrix A,
	vector v, const enum reduction_op, const enum unary_op) */
ok_status linalg_matrix_row_squares(const enum CBLAS_TRANSPOSE t,
	const matrix *A, vector *v);
ok_status linalg_matrix_broadcast_vector(matrix *A, const vector *v,
	const enum OPTKIT_TRANSFORM operation, const enum CBLAS_SIDE side);
ok_status linalg_matrix_reduce_indmin(indvector *indices, vector *minima,
	const matrix *A, const enum CBLAS_SIDE side);
ok_status linalg_matrix_reduce_min(vector *minima, const matrix *A,
	const enum CBLAS_SIDE side);
ok_status linalg_matrix_reduce_max(vector *maxima, const matrix *A,
	const enum CBLAS_SIDE side);

#ifdef __cplusplus
}
#endif

#endif  /* OPTKIT_LINSYS_DENSE_H_ */
