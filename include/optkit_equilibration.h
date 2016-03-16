#ifndef OPTKIT_EQUILIBRATION_H_
#define OPTKIT_EQUILIBRATION_H_

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

void equillib_version(int * maj, int * min, int * change, int * status);

void sinkhorn_knopp(void * linalg_handle, ok_float * A_in, matrix * A_out,
	vector * d, vector *e, enum CBLAS_ORDER ord);
void regularized_sinkhorn_knopp(void * linalg_handle, ok_float * A_in,
	matrix * A_out, vector * d, vector *e, enum CBLAS_ORDER ord);
void dense_l2(void * linalg_handle, ok_float * A_in, matrix * A_out,
	vector * d, vector * e, enum CBLAS_ORDER ord);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_EQUILIBRATION_H_ */