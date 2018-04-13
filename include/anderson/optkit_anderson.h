#ifndef OPTKIT_ANDERSON_H_
#define OPTKIT_ANDERSON_H_

#include "optkit_lapack.h"
#include "optkit_anderson_reductions.h"

#define kANDERSON_SILENCE_CHOLESKY 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anderson_accelerator{
	size_t vector_dim, lookback_dim;
	matrix *F, *G, *F_gram;
	vector *alpha, *ones;
	ok_float mu_regularization;
	size_t iter;
	void *blas_handle;
	void *lapack_handle;
} anderson_accelerator;

typedef struct fused_accelerator{
	size_t vector_dim, lookback_dim, n_blocks;
	matrix *F, *G, *F_gram;
	vector *w, *alpha, *ones;
	ok_float mu_regularization;
	size_t iter;
	void *blas_handle;
	void *lapack_handle;
} fused_accelerator;


/* ANDERSON, EXPLICITLY CONSTRAINED FORMULATION */
ok_status anderson_update_F_x(matrix *F, vector *x, size_t index);
ok_status anderson_update_F_g(matrix *F, vector *gx, size_t index);
ok_status anderson_update_G(matrix *G, vector *gx, size_t index);
ok_status anderson_regularized_gram(void *blas_hdl, matrix *F, matrix *F_gram,
	ok_float mu);
ok_status anderson_autoregularize(vector *F_gram_diag, ok_float *mu_auto);

ok_status anderson_solve(void *blas_hdl, void *lapack_hdl, matrix *F,
	matrix *F_gram, vector *alpha, const vector *ones, ok_float mu);
ok_status anderson_mix(void *blas_hdl, matrix *G, vector *alpha,
	vector *x);
ok_status anderson_accelerate_template(void *blas_hdl, void *lapack_hdl,
	matrix *F, matrix *G, matrix *F_gram, vector *alpha, const vector *ones,
	ok_float mu, size_t *iter, vector *iterate, vector *x_reduced,
	size_t n_reduction,
	ok_status (* x_reduction)(vector *x_rdx, vector *x, size_t n_rdx));

/* UNMODIFIED STATE VARIABLE */
ok_status anderson_accelerator_init(anderson_accelerator *aa,
	size_t vector_dim, size_t lookback_dim);
ok_status anderson_accelerator_free(anderson_accelerator *aa);
ok_status anderson_set_x0(anderson_accelerator *aa, vector *x_initial);
ok_status anderson_accelerate(anderson_accelerator *aa, vector *x);

/* SUMMED STATE VARIABLE BLOCKS */
ok_status anderson_fused_accelerator_init(fused_accelerator *aa,
	size_t vector_dim, size_t lookback_dim, size_t n_blocks);
ok_status anderson_fused_accelerator_free(fused_accelerator *aa);
ok_status anderson_fused_set_x0(fused_accelerator *aa, vector *x_initial);
ok_status anderson_fused_accelerate(fused_accelerator *aa, vector *x);


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_ANDERSON_H_ */
