#ifndef OPTKIT_EQUILIBRATION_H_
#define OPTKIT_EQUILIBRATION_H_

#include "optkit_dense.h"
#ifndef OPTKIT_NO_OPERATOR_EQUIL
#include "optkit_operator_transforms.h"
#include "optkit_operator_dense.h"
#include "optkit_operator_sparse.h"
#endif /* ndef OPTKIT_NO_OPERATOR_EQUIL */

#ifdef __cplusplus
extern "C" {
#endif

ok_status regularized_sinkhorn_knopp(void *blas_handle, ok_float *A_in,
	matrix *A_out, vector *d, vector *e, enum CBLAS_ORDER ord);

#ifndef OPTKIT_NO_OPERATOR_EQUIL
ok_status operator_regularized_sinkhorn(void *blas_handle,
	abstract_operator *A, vector *d, vector *e, const ok_float pnorm);
ok_status operator_equilibrate(void *blas_handle, abstract_operator *A,
	vector *d, vector *e, const ok_float pnorm);
ok_status operator_estimate_norm(void *blas_handle, abstract_operator *A,
	ok_float *norm_est);
#endif /* ndef OPTKIT_NO_OPERATOR_EQUIL */

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_EQUILIBRATION_H_ */
