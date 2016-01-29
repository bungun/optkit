#ifndef OPTKIT_EQUILIBRATION_H_GUARD
#define OPTKIT_EQUILIBRATION_H_GUARD

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

void equillib_version(int * maj, int * min, int * change, int * status);


void sinkhorn_knopp(void * linalg_handle, ok_float * A_in, matrix * A_out, 
	vector * d, vector *e, CBLAS_ORDER_t ord);
void dense_l2(void * linalg_handle, ok_float * A_in, matrix * A_out, 
	vector * d, vector * e, CBLAS_ORDER_t ord);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_EQUILIBRATION_H_GUARD */