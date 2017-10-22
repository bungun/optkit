#ifndef OPTKIT_ANDERSON_H_
#define OPTKIT_ANDERSON_H_

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	size_t vector_dim, lookback_dim;
	matrix * F, * G, * F_gram;	/* pre-allocated matrices */
	vector * f, * g, * diag; 	/* views of above matrices */
	vector * alpha, * ones; 	/* vectors in R^{lookback_dim + 1} */
	ok_float mu_regularization;
	size_t iter;
	void * linalg_handle;
} anderson_accelerator;

anderson_accelerator * anderson_accelerator_init(vector * x_initial,
	size_t lookback_dim);
ok_status anderson_accelerator_free(anderson_accelerator * aa);
ok_status anderson_update_F_x(anderson_accelerator * aa, matrix * F, vector * x,
	size_t index);
ok_status anderson_update_F_g(anderson_accelerator * aa, matrix * F,
	vector * gx, size_t index);
ok_status anderson_update_G(anderson_accelerator * aa, matrix * G, vector * gx,
	size_t index);
ok_status anderson_regularized_gram(anderson_accelerator * aa, matrix * F,
	matrix * F_gram, ok_float mu);
ok_status anderson_solve(anderson_accelerator *aa, matrix * F, vector * alpha,
	ok_float mu);
ok_status anderson_mix(anderson_accelerator *aa, matrix * G, vector * alpha,
	vector * x);
ok_status anderson_accelerate(anderson_accelerator * aa, vector * x);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_ANDERSON_H_ */