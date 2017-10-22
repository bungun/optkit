#include "optkit_anderson_acceleration.h"

#ifdef __cplusplus
extern "C" {
#endif

anderson_accelerator * accelerator_init(vector * x_initial, size_t lookback_dim){
	ok_status err = OPTKIT_SUCCESS;
	anderson_accelerator * aa = OK_NULL;
	ok_alloc(aa, sizeof(*aa));
	OK_CHECK_PTR(aa);
	OK_CHECK_VECTOR(x_initial)

	if (lookback_dim > vector_dim)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	aa.vector_dim = x_initial->size;
	aa.lookback_dim = lookback_dim;
	aa.mu_regularization = (ok_float) 0.01;
	aa.fixed_size = 1u;
	aa.iter = 0;

	ok_alloc(aa->F, sizeof(*aa->F));
	OK_MAX_ERR( err, matrix_calloc(aa->F, vector_dim, lookback_dim + 1) );
	ok_alloc(aa->X, sizeof(*aa->X));
	OK_MAX_ERR( err, matrix_calloc(aa->X, vector_dim, lookback_dim + 1) );
	ok_alloc(aa->F_gram, sizeof(*aa->F_gram));
	OK_MAX_ERR( err, matrix_calloc(aa->F_gram, lookback_dim + 1,
		lookback_dim + 1) );

	ok_alloc(aa->f, sizeof(*aa->f));
	ok_alloc(aa->g, sizeof(*aa->g));
	ok_alloc(aa->diag, sizeof(**aa->diag));

	ok_alloc(aa->alpha, sizeof(*aa->alpha));
	OK_MAX_ERR( err, vector_calloc(aa->alpha, lookback_dim + 1) );
	ok_alloc(aa->ones, sizeof(*aa->ones));
	OK_MAX_ERR( err, vector_calloc(aa->ones, lookback_dim + 1) );

	OK_MAX_ERR( err, blas_make_handle(&(aa->blas_handle)) );

	if (err) {
		OK_MAX_ERR( err, accelerator_free(aa) );
		aa = OK_NULL;
	} else {
		/* update f_0, g_0 */
		OK_CHECK_ERR( accelerator_update_F_x(aa, aa->F, x_initial, 0) );
		OK_CHECK_ERR( accelerator_update_G(aa, aa->G, x_initial, 0) );

		/* initialize aa->ones to 1 vector */
		OK_CHECK_ERR( vector_set_all(aa->ones, kOne) );
	}
	return aa;
}

ok_status accelerator_free(anderson_accelerator * aa){
	ok_status err = OK_SCAN_ERR( blas_destroy_handle(aa->blas_handle) );
	OK_MAX_ERR( err, matrix_free(aa->F) );
	OK_MAX_ERR( err, matrix_free(aa->X) );
	OK_MAX_ERR( err, matrix_free(aa->F_gram) );

	OK_MAX_ERR( err, vector_free(aa->alpha) );
	OK_MAX_ERR( err, vector_free(aa->ones) );

	ok_free(aa->F);
	ok_free(aa->X);
	ok_free(aa->F_gram);

	ok_free(aa->f);
	ok_free(aa->g);
	ok_free(aa->diag);

	ok_free(aa->alpha);
	ok_free(aa->ones);

	ok_free(aa);

	return err;
};

ok_status anderson_update_F_x(anderson_accelerator * aa, matrix * F, vector * x,
	size_t index){
	OK_RETURNIF_ERR( matrix_column(aa->f, aa->F, index) );
	OK_RETURNIF_ERR( blas_axpy(aa->blas_handle, kOne, x, aa->f) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_update_F_g(anderson_accelerator * aa, matrix * F, vector * gx,
	size_t index){
	OK_RETURNIF_ERR( matrix_column(aa->f, aa->F, index) );
	OK_RETURNIF_ERR( vector_copy(aa->f, gx) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_update_G(anderson_accelerator * aa, matrix * G, vector * gx,
	size_t index){
	OK_RETURNIF_ERR( matrix_column(aa->g, aa->G, index) );
	OK_RETURNIF_ERR( vector_copy(aa->g, gx) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_regularized_gram(anderson_accelerator * aa, matrix * F,
	matrix * F_gram, ok_float mu){
	/* F_gram = F'F */
	OK_RETURNIF_ERR( blas_gemm(aa->blas_handle, CblasTrans, CblasNoTrans,
		kOne, F, F, kZero, F_gram) )
	// TODO: consider sryk instead of gemm:
	//	OK_RETURNIF_ERR( blas_syrk(aa->blas_handle, CblasLower,
	//		CblasTrans, kOne, aa->F, kZero, aa->F_gram) )

	/* F_gram = F'F + \sqrt(mu) I */
	OK_RETURNIF_ERR( matrix_diagonal(aa->diag, F_gram) );
	OK_RETURNIF_ERR( vector_add_constant(aa->diag, MATH(sqrt)(aa->mu)) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_solve(anderson_accelerator *aa, matrix * F, vector * alpha,
	ok_float mu){
	ok_float denominator = kOne;

	/* F_gram = F'F + \sqrt(mu)I  */
	OK_RETURNIF_ERR( anderson_regularized_gram(aa, aa->F, aa->F_gram, mu) );

	/* LL' = F_gram */
	OK_RETURNIF_ERR( linalg_cholesky_decomp(aa->blas_handle, aa->F_gram) );

	/* alpha_hat = F_gram^{-1} 1 */
	OK_RETURNIF_ERR( vector_set_all(aa->alpha, kOne) )
	OK_RETURNIF_ERR( linalg_cholesky_svx(aa->blas_handle,
		aa->F_gram, aa->alpha))

	/* denom = 1'alpha_hat */
	OK_RETURNIF_ERR( blas_dot(aa->blas_handle, aa->ones, alpha,
		&denominator) );

	/*
	 * alpha = alpha_hat / denom
	 * 	 = alpha_hat / 1'alpha_hat
	 *	 = (F'F + \sqrt(mu)I)^{-1}1 / 1'(F'F + \sqrt(mu)I)^{-1}1
	 */
	OK_RETURNIF_ERR( vector_scale(alpha, 1. / denominator) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_mix(anderson_accelerator *aa, matrix * G, vector * alpha,
	vector * x){
	OK_RETURNIF_ERR( blas_axpy(aa->blas_handle, CblasNoTrans, kOne, G,
		alpha, kZero, x) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_accelerate(anderson_accelerator * aa, vector * x){
	size_t index = aa->iter % (aa->lookback_dim + 1);
	ok_float denominator, x_scale;

	/* CHECK POINTERS, COMPATIBILITY OF X */
	OK_CHECK_PTR(aa);
	OK_CHECK_VECTOR(x);
	if (aa->vector_dim != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	/* UPDATE f_i, g_i */
	OK_RETURNIF_ERR( anderson_update_F_g(aa, aa->F, x, index) );
	OK_RETURNIF_ERR( anderson_update_G(aa, aa->G, x, index) );

	/* CHECK ITERATION >= LOOKBACK */
	if (aa->iter < aa->lookback_dim)
		return;

	/* SOLVE argmin_\alpha ||F \alpha||_2 s.t. 1'\alpha = 1 */
	OK_RETURNIF_ERR( anderson_solve(aa, aa->F, aa->alpha,
		aa->mu_regularization) );

	/* x = G \alpha */
	OK_RETURNIF_ERR( anderson_mix(aa, aa->G, aa->alpha, x) );

	// UPDATE COUNTER and START BUILDING f_{i+1}
	aa->iter += 1;
	OK_RETURNIF_ERR( anderson_update_F_x(aa, aa->F, x, index) );

	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif