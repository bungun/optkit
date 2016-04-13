#ifndef OPTKIT_LINSYS_DENSE_H_
#define OPTKIT_LINSYS_DENSE_H_

#include "optkit_matrix.hpp"
#include "optkit_blas.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void denselib_version(int * maj, int * min, int * change, int * status);

/* LINEAR ALGEBRA routines */
void linalg_cholesky_decomp(void * linalg_handle, matrix * A);
void linalg_cholesky_svx(void * linalg_handle, const matrix * L, vector * x);
void linalg_diag_gramian(void * linalg_handle, const matrix * A, vector * v);
void linalg_matrix_broadcast_vector(void * linalg_handle, matrix * A,
	const vector * v, const enum OPTKIT_TRANSFORM operation,
	const enum CBLAS_SIDE side);
void linalg_matrix_reduce_indmin(void * linalg_handle, size_t * indices,
	vector * minima, matrix * A, const enum CBLAS_SIDE side);
void linalg_matrix_reduce_min(void * linalg_handle, vector * minima,
	matrix * A, const enum CBLAS_SIDE side);
void linalg_matrix_reduce_max(void * linalg_handle, vector * maxima,
	matrix * A, const enum CBLAS_SIDE side);

/* device reset */
ok_status ok_device_reset(void);

#ifdef __cplusplus
}
#endif

#endif  /* OPTKIT_LINSYS_DENSE_H_ */
