#include "optkit_anderson.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status anderson_accelerator_init(anderson_accelerator * aa,
	size_t vector_dim, size_t lookback_dim)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(aa);
	if (aa->F)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	if (lookback_dim > vector_dim)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

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

	ok_alloc(aa->f, sizeof(*aa->f));
	ok_alloc(aa->g, sizeof(*aa->g));
	ok_alloc(aa->diag, sizeof(*aa->diag));

	ok_alloc(aa->alpha, sizeof(*aa->alpha));
	OK_CHECK_ERR( err, vector_calloc(aa->alpha, lookback_dim + 1) );
	ok_alloc(aa->ones, sizeof(*aa->ones));
	OK_CHECK_ERR( err, vector_calloc(aa->ones, lookback_dim + 1) );

	OK_CHECK_ERR( err, blas_make_handle(&(aa->linalg_handle)) );

	/* initialize aa->ones to 1 vector */
	OK_CHECK_ERR( err, vector_set_all(aa->ones, kOne) );

	if (err)
		OK_MAX_ERR( err, anderson_accelerator_free(aa) );

	return err;
}

ok_status anderson_accelerator_free(anderson_accelerator * aa)
{
	ok_status err = OK_SCAN_ERR( blas_destroy_handle(aa->linalg_handle) );
	OK_MAX_ERR( err, matrix_free(aa->F) );
	OK_MAX_ERR( err, matrix_free(aa->G) );
	OK_MAX_ERR( err, matrix_free(aa->F_gram) );

	OK_MAX_ERR( err, vector_free(aa->alpha) );
	OK_MAX_ERR( err, vector_free(aa->ones) );

	ok_free(aa->F);
	ok_free(aa->G);
	ok_free(aa->F_gram);

	ok_free(aa->f);
	ok_free(aa->g);
	ok_free(aa->diag);

	ok_free(aa->alpha);
	ok_free(aa->ones);

	return err;
}

ok_status anderson_set_x0(anderson_accelerator * aa, vector * x_initial){
	ok_status err = OPTKIT_SUCCESS;
	if (!x_initial || !x_initial->data)
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	else
		if (x_initial->size != aa->vector_dim)
			err = OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	/*
	 * update F, G:
	 *	F = [-x0 0 ... 0]
	 * 	G = [0 0 ... 0]
	 */
	OK_CHECK_ERR( err, anderson_update_F_x(aa, aa->F, x_initial, 0) );
	return err;
}

/* At index i, set f_i = -x */
ok_status anderson_update_F_x(anderson_accelerator * aa, matrix * F, vector * x,
	size_t index)
{
	OK_RETURNIF_ERR( matrix_column(aa->f, F, index) );
	OK_RETURNIF_ERR( vector_memcpy_vv(aa->f, x) );
	OK_RETURNIF_ERR( vector_scale(aa->f, -kOne) );
	return OPTKIT_SUCCESS;
}

/* At index i, perform f_i += g(x); call after anderson_update_F_x */
ok_status anderson_update_F_g(anderson_accelerator * aa, matrix * F,
	vector * gx, size_t index)
{
	OK_RETURNIF_ERR( matrix_column(aa->f, F, index) );
	OK_RETURNIF_ERR( vector_add(aa->f, gx) );
	return OPTKIT_SUCCESS;
}

/* At index i, set g_i = x */
ok_status anderson_update_G(anderson_accelerator * aa, matrix * G, vector * gx,
	size_t index)
{
	OK_RETURNIF_ERR( matrix_column(aa->g, G, index) );
	OK_RETURNIF_ERR( vector_memcpy_vv(aa->g, gx) );
	return OPTKIT_SUCCESS;
}

ok_status anderson_regularized_gram(anderson_accelerator * aa, matrix * F,
	matrix * F_gram, ok_float mu)
{
	ok_status err = OPTKIT_SUCCESS;

	/* F_gram = F'F */
	OK_CHECK_ERR( err, blas_gemm(aa->linalg_handle, CblasTrans, CblasNoTrans,
		kOne, F, F, kZero, F_gram) );
	// TODO: consider sryk instead of gemm:
	//	OK_CHECK_ERR( err, blas_syrk(aa->linalg_handle, CblasLower,
	//		CblasTrans, kOne, aa->F, kZero, aa->F_gram) )

	/* F_gram = F'F + \sqrt(mu) I */
	OK_CHECK_ERR( err, matrix_diagonal(aa->diag, F_gram) );
	OK_CHECK_ERR( err, vector_add_constant(aa->diag, MATH(sqrt)(
		aa->mu_regularization)) );
	return err;
}

ok_status anderson_solve(anderson_accelerator *aa, matrix * F, vector * alpha,
	ok_float mu)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_float denominator = 1.;

	/* F_gram = F'F + \sqrt(mu)I  */
	OK_CHECK_ERR( err, anderson_regularized_gram(aa, F, aa->F_gram, mu) );

	/* LL' = F_gram */
	OK_CHECK_ERR( err, linalg_cholesky_decomp(aa->linalg_handle, aa->F_gram) );

	/* alpha_hat = F_gram^{-1} 1 */
	OK_CHECK_ERR( err, vector_set_all(alpha, kOne) );
	OK_CHECK_ERR( err, linalg_cholesky_svx(aa->linalg_handle,
		aa->F_gram, alpha));

	/* denom = 1'alpha_hat */
	OK_CHECK_ERR( err, blas_dot(aa->linalg_handle, aa->ones, alpha,
		&denominator) );

	/*
	 * alpha = alpha_hat / denom
	 * 	 = alpha_hat / 1'alpha_hat
	 *	 = (F'F + \sqrt(mu)I)^{-1}1 / 1'(F'F + \sqrt(mu)I)^{-1}1
	 */
	OK_CHECK_ERR( err, vector_scale(alpha, kOne / denominator) );
	return err;
}

ok_status anderson_mix(anderson_accelerator *aa, matrix * G, vector * alpha,
	vector * x)
{
	OK_RETURNIF_ERR( blas_gemv(aa->linalg_handle, CblasNoTrans, kOne, G,
		alpha, kZero, x) );
	return OPTKIT_SUCCESS;
}

/*
 * Perform one iteration of Anderson Acceleration for some input taken
 * to be an iterate of some fixed-point algorithm given by x_{k+1} = G(x_k)
 *
 * For iteration k, inputs represent:
 * 	anderson_accelerator * aa: an object containing the state of the
 *		algorithm, namely the last ``aa->lookback_dim`` iterates and
 *		residuals (given by the columns of matrices ``aa->G`` and
 *		``aa->F``, respectively), as well as the iteration counter.
 *	vector * x: assumed to represent fixed-point iterate G(x_k)
 *
 * The output x_{k+1} is written over the input G(x_k).
 * If k < aa->lookback_dim, the input G(x_k) is passed to the output x_{k+1}.
 * If k >= aa->lookback_dim, Anderson Acceleration is performed to obtain a
 * 	linear combination of the iterates stored in the columns of ``aa->G``,
 *	with the weights ``aa->alpha`` chosen to minimize the norm of the same
 *	linear combination of the residuals stored in the columns of ``aa->F``.
 */
ok_status anderson_accelerate(anderson_accelerator * aa, vector * x)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t index = aa->iter % (aa->lookback_dim + 1);
	size_t next_index = (aa->iter + 1) % (aa->lookback_dim + 1);

	/* CHECK POINTERS, COMPATIBILITY OF X */
	OK_CHECK_PTR(aa);
	OK_CHECK_VECTOR(x);
	if (aa->vector_dim != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	/*
	 * UPDATE G, COMPLETE F UPDATE:
	 *	g_i = G(x_i)
	 *	f_i +=  G(x_i), should now contain G(x_i) - x_i
	 */
	OK_CHECK_ERR( err, anderson_update_F_g(aa, aa->F, x, index) );
	OK_CHECK_ERR( err, anderson_update_G(aa, aa->G, x, index) );

	/* CHECK ITERATION >= LOOKBACK */
	if (aa->iter >= aa->lookback_dim){
		/* SOLVE argmin_\alpha ||F \alpha||_2 s.t. 1'\alpha = 1 */
		OK_CHECK_ERR( err, anderson_solve(aa, aa->F, aa->alpha,
			aa->mu_regularization) );

		/* x = G \alpha */
		OK_CHECK_ERR( err, anderson_mix(aa, aa->G, aa->alpha, x) );
	}

	/*
	 * START BUILDING F UPDATE
	 *	f_{i + 1} = -x_{i + 1}
	 */
	OK_CHECK_ERR( err, anderson_update_F_x(aa, aa->F, x, next_index) );

	aa->iter += 1;
	return err;
}

#ifdef __cplusplus
}
#endif
