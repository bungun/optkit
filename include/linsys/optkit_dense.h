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

/* device reset */
ok_status ok_device_reset(void);

#ifdef __cplusplus
}
#endif

#endif  /* OPTKIT_LINSYS_DENSE_H_ */
