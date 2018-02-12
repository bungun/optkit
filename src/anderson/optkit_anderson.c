#include "optkit_anderson.h"

#ifdef __cplusplus
extern "C" {
#endif

/* At index i, set f_i = -x */
ok_status anderson_update_F_x(matrix *F, vector *x, size_t index)
{
	vector fcol = (vector){OK_NULL};
	OK_RETURNIF_ERR( matrix_column(&fcol, F, index) );
	OK_RETURNIF_ERR( vector_memcpy_vv(&fcol, x) );
	OK_RETURNIF_ERR( vector_scale(&fcol, -kOne) );
	return OPTKIT_SUCCESS;
}

/* At index i, perform f_i += g(x); call after anderson_update_F_x */
ok_status anderson_update_F_g(matrix *F, vector *gx, size_t index)
{
	vector fcol = (vector){OK_NULL};
	OK_RETURNIF_ERR( matrix_column(&fcol, F, index) );
	OK_RETURNIF_ERR( vector_add(&fcol, gx) );
	return OPTKIT_SUCCESS;
}

/* At index i, set g_i = x */
ok_status anderson_update_G(matrix *G, vector *gx, size_t index)
{
	vector gcol = (vector){OK_NULL};
	OK_RETURNIF_ERR( matrix_column(&gcol, G, index) );
	OK_RETURNIF_ERR( vector_memcpy_vv(&gcol, gx) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_regularized_gram(void *blas_handle, matrix *F,
	matrix *F_gram, ok_float mu)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_float sqrt_mu = kZero;
	vector diag = (vector){OK_NULL};

	/* F_gram = F'F */
	OK_CHECK_ERR( err, blas_gemm(blas_handle, CblasTrans, CblasNoTrans,
		kOne, F, F, kZero, F_gram) );
	// TODO: consider sryk instead of gemm:
	//	OK_CHECK_ERR( err, blas_syrk(blas_handle, CblasLower,
	//		CblasTrans, kOne, F, kZero, F_gram) )

	/* F_gram = F'F + \sqrt(mu) I */
	OK_CHECK_ERR( err, matrix_diagonal(&diag, F_gram) );
	if (mu > 0) {
		sqrt_mu = MATH(sqrt)(mu);
		OK_CHECK_ERR( err, anderson_autoregularize(&diag, &sqrt_mu) );
		OK_CHECK_ERR( err, vector_add_constant(&diag, sqrt_mu) );
	}
	return err;
}

ok_status anderson_autoregularize(vector *F_gram_diag, ok_float *mu_auto)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_float mu_requested;
	OK_CHECK_PTR(mu_auto);
	OK_CHECK_VECTOR(F_gram_diag);

	mu_requested = *mu_auto;
	OK_CHECK_ERR( err, vector_max(F_gram_diag, mu_auto) );
	*mu_auto *= mu_requested;
	return err;
}

ok_status anderson_solve(void *hdl, matrix *F, matrix *F_gram, vector *alpha,
	const vector *ones, ok_float mu)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_status cholesky_err = OPTKIT_SUCCESS;
	ok_float denominator = kOne;

	/* F_gram = F'F + \sqrt(mu)I  */
	OK_CHECK_ERR( err, anderson_regularized_gram(hdl, F, F_gram, kZero) );

	/* LL' = F_gram */
	if (!err)
		cholesky_err = linalg_cholesky_decomp_flagged(hdl, F_gram,
			kANDERSON_SILENCE_CHOLESKY);
	if (!kANDERSON_SILENCE_CHOLESKY)
		OK_SCAN_ERR(cholesky_err);

	if (cholesky_err && mu > 0) {
		OK_CHECK_ERR( err, anderson_regularized_gram(hdl, F, F_gram, mu) );
		OK_CHECK_ERR( err, linalg_cholesky_decomp(hdl, F_gram) );
	} else if (cholesky_err) {
		err = cholesky_err > err ? cholesky_err : err;
	}

	/* alpha_hat = F_gram^{-1} 1 */
	OK_CHECK_ERR( err, vector_set_all(alpha, kOne) );
	OK_CHECK_ERR( err, linalg_cholesky_svx(hdl, F_gram, alpha));

	/* denom = 1'alpha_hat */
	OK_CHECK_ERR( err, blas_dot(hdl, ones, alpha, &denominator) );

	/*
	 * alpha = alpha_hat / denom
	 * 	 = alpha_hat / 1'alpha_hat
	 *	 = (F'F + \sqrt(mu)I)^{-1}1 / 1'(F'F + \sqrt(mu)I)^{-1}1
	 */
	OK_CHECK_ERR( err, vector_scale(alpha, kOne / denominator) );
	return err;
}

ok_status anderson_mix(void *hdl, matrix *G, vector *alpha, vector *x)
{
	return OK_SCAN_ERR( blas_gemv(hdl, CblasNoTrans, kOne, G, alpha, kZero, x) );
}

/*
 * Perform one iteration of Anderson Acceleration for some input taken
 * to be an iterate of some fixed-point algorithm given by x_{k+1} = G(x_k)
 *
 * For iteration k, inputs represent:
 * 	anderson_accelerator *aa: an object containing the state of the
 *		algorithm, namely the last ``aa->lookback_dim`` iterates and
 *		residuals (given by the columns of matrices ``aa->G`` and
 *		``aa->F``, respectively), as well as the iteration counter.
 *	vector *x: assumed to represent fixed-point iterate G(x_k)
 *
 * The output x_{k+1} is written over the input G(x_k).
 * If k < aa->lookback_dim, the input G(x_k) is passed to the output x_{k+1}.
 * If k >= aa->lookback_dim, Anderson Acceleration is performed to obtain a
 * 	linear combination of the iterates stored in the columns of ``aa->G``,
 *	with the weights ``aa->alpha`` chosen to minimize the norm of the same
 *	linear combination of the residuals stored in the columns of ``aa->F``.
 */

ok_status anderson_accelerate_template(void *hdl, matrix *F, matrix *G,
	matrix *F_gram, vector *alpha, const vector *ones, ok_float mu,
	size_t *iter, vector *iterate, vector *x_reduced, size_t n_reduction,
	ok_status (* x_reduction)(vector *x_rdx, vector *x, size_t n_rdx))
{
	ok_status err = OPTKIT_SUCCESS;
	ok_status cholesky_err = OPTKIT_SUCCESS;
	size_t lookback_dim, index, next_index;

	/* CHECK POINTERS, COMPATIBILITY OF X */
	OK_CHECK_MATRIX(F);
	OK_CHECK_MATRIX(G);
	OK_CHECK_MATRIX(F_gram);
	OK_CHECK_VECTOR(alpha);
	OK_CHECK_VECTOR(ones);
	OK_CHECK_PTR(iter);
	OK_CHECK_VECTOR(iterate);
	OK_CHECK_VECTOR(x_reduced);
	if (G->size1 != iterate->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	lookback_dim = F->size2 - 1;
	index = *iter % (F->size2);
	next_index = (*iter + 1) % (F->size2);

	/*
	 * UPDATE G, COMPLETE F UPDATE:
	 *	g_i = G(x_i)
	 *	f_i +=  G(x_i), should now contain G(x_i) - x_i
	 */
	OK_CHECK_ERR( err, x_reduction(x_reduced, iterate, n_reduction) );
	OK_CHECK_ERR( err, anderson_update_F_g(F, x_reduced, index) );
	OK_CHECK_ERR( err, anderson_update_G(G, iterate, index) );

	/* CHECK ITERATION >= LOOKBACK */
	if (*iter >= lookback_dim){
		/* SOLVE argmin_\alpha ||F \alpha||_2 s.t. 1'\alpha = 1 */
		OK_CHECK_ERR( cholesky_err, anderson_solve(hdl, F, F_gram,
			alpha, ones, mu) );

		/* x = G \alpha */
		if (!cholesky_err)
			OK_CHECK_ERR( err, anderson_mix(hdl, G, alpha, iterate) );
	}

	/*
	 * START BUILDING F UPDATE
	 *	f_{i + 1} = -x_{i + 1}
	 */
	OK_CHECK_ERR( err, x_reduction(x_reduced, iterate, n_reduction) );
	OK_CHECK_ERR( err, anderson_update_F_x(F, x_reduced, next_index) );

	*iter += 1;
	return err;
}

ok_status anderson_accelerator_init(anderson_accelerator *aa,
	size_t vector_dim, size_t lookback_dim)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(aa);
	if (aa->F)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	if (lookback_dim > vector_dim)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
		// lookback_dim = vector_dim;

	aa->vector_dim = vector_dim;
	aa->lookback_dim = lookback_dim;
	aa->mu_regularization = (ok_float) 0.01;
	aa->iter = 0;

	ok_alloc(aa->F, sizeof(*aa->F));
	OK_CHECK_ERR( err, matrix_calloc(aa->F, vector_dim, lookback_dim + 1,
		CblasColMajor) );
	ok_alloc(aa->G, sizeof(*aa->G));
	OK_CHECK_ERR( err, matrix_calloc(aa->G, vector_dim, lookback_dim + 1,
		CblasColMajor) );
	ok_alloc(aa->F_gram, sizeof(*aa->F_gram));
	OK_CHECK_ERR( err, matrix_calloc(aa->F_gram, lookback_dim + 1,
		lookback_dim + 1, CblasColMajor) );

	ok_alloc(aa->alpha, sizeof(*aa->alpha));
	OK_CHECK_ERR( err, vector_calloc(aa->alpha, lookback_dim + 1) );
	ok_alloc(aa->ones, sizeof(*aa->ones));
	OK_CHECK_ERR( err, vector_calloc(aa->ones, lookback_dim + 1) );

	OK_CHECK_ERR( err, blas_make_handle(&(aa->blas_handle)) );

	/* initialize aa->ones to 1 vector */
	OK_CHECK_ERR( err, vector_set_all(aa->ones, kOne) );

	if (err)
		OK_MAX_ERR( err, anderson_accelerator_free(aa) );

	return err;
}

ok_status anderson_accelerator_free(anderson_accelerator *aa)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(aa);
	OK_MAX_ERR( err, blas_destroy_handle(aa->blas_handle) );
	OK_MAX_ERR( err, matrix_free(aa->F) );
	OK_MAX_ERR( err, matrix_free(aa->G) );
	OK_MAX_ERR( err, matrix_free(aa->F_gram) );

	OK_MAX_ERR( err, vector_free(aa->alpha) );
	OK_MAX_ERR( err, vector_free(aa->ones) );

	ok_free(aa->F);
	ok_free(aa->G);
	ok_free(aa->F_gram);

	ok_free(aa->alpha);
	ok_free(aa->ones);

	return err;
}

ok_status anderson_set_x0(anderson_accelerator *aa, vector *x_initial)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_VECTOR(x_initial);
	if (x_initial->size != aa->vector_dim)
		err = OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	/*
	 * update F, G:
	 *	F = [-x0 0 ... 0]
	 * 	G = [0 0 ... 0]
	 */
	OK_CHECK_ERR( err, anderson_update_F_x(aa->F, x_initial, 0) );
	return err;
}

ok_status anderson_accelerate(anderson_accelerator *aa, vector *x)
{
	OK_CHECK_PTR(aa);
	return OK_SCAN_ERR( anderson_accelerate_template(
		aa->blas_handle, aa->F, aa->G, aa->F_gram, aa->alpha,
		aa->ones, aa->mu_regularization, &aa->iter, x, x, (size_t) 0,
		anderson_reduce_null) );
}

#ifdef __cplusplus
}
#endif
