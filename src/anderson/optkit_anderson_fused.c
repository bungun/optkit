#include "optkit_anderson.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status anderson_fused_accelerator_init(fused_accelerator *aa,
	size_t vector_dim, size_t lookback_dim, size_t n_blocks)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(aa);
	if (aa->F)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	if (lookback_dim > vector_dim)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
		// lookback_dim = vector_dim;
	if (vector_dim % n_blocks != 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

	aa->vector_dim = vector_dim;
	aa->lookback_dim = lookback_dim;
	aa->n_blocks = n_blocks;
	aa->mu_regularization = (ok_float) 0.01;
	aa->iter = 0;

	ok_alloc(aa->F, sizeof(*aa->F));
	OK_CHECK_ERR( err, matrix_calloc(aa->F, vector_dim / n_blocks,
		lookback_dim + 1, CblasColMajor) );
	ok_alloc(aa->G, sizeof(*aa->G));
	OK_CHECK_ERR( err, matrix_calloc(aa->G, vector_dim,
		lookback_dim + 1, CblasColMajor) );
	ok_alloc(aa->F_gram, sizeof(*aa->F_gram));
	OK_CHECK_ERR( err, matrix_calloc(aa->F_gram, lookback_dim + 1,
		lookback_dim + 1, CblasColMajor) );

	ok_alloc(aa->w, sizeof(*aa->w));
	OK_CHECK_ERR( err, vector_calloc(aa->w, vector_dim / n_blocks) );
	ok_alloc(aa->alpha, sizeof(*aa->alpha));
	OK_CHECK_ERR( err, vector_calloc(aa->alpha, lookback_dim + 1) );
	ok_alloc(aa->ones, sizeof(*aa->ones));
	OK_CHECK_ERR( err, vector_calloc(aa->ones, lookback_dim + 1) );

	OK_CHECK_ERR( err, blas_make_handle(&(aa->blas_handle)) );
	OK_CHECK_ERR( err, lapack_make_handle(&(aa->lapack_handle)) );

	/* initialize aa->ones to 1 vector */
	OK_CHECK_ERR( err, vector_set_all(aa->ones, kOne) );

	if (err)
		OK_MAX_ERR( err, anderson_fused_accelerator_free(aa) );

	return err;
}

ok_status anderson_fused_accelerator_free(fused_accelerator *aa)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(aa);
	OK_MAX_ERR( err, blas_destroy_handle(aa->blas_handle) );
	OK_MAX_ERR( err, lapack_destroy_handle(aa->lapack_handle) );
	OK_MAX_ERR( err, matrix_free(aa->F) );
	OK_MAX_ERR( err, matrix_free(aa->G) );
	OK_MAX_ERR( err, matrix_free(aa->F_gram) );

	OK_MAX_ERR( err, vector_free(aa->w) );
	OK_MAX_ERR( err, vector_free(aa->alpha) );
	OK_MAX_ERR( err, vector_free(aa->ones) );

	ok_free(aa->F);
	ok_free(aa->G);
	ok_free(aa->F_gram);

	ok_free(aa->w);
	ok_free(aa->alpha);
	ok_free(aa->ones);
	return err;
}


ok_status anderson_fused_set_x0(fused_accelerator *aa, vector *x_initial)
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
	OK_CHECK_ERR( err, anderson_sum_blocks(aa->w, x_initial, aa->n_blocks) );
	OK_CHECK_ERR( err, anderson_update_F_x(aa->F, aa->w, 0) );
	return err;
}

ok_status anderson_fused_accelerate(fused_accelerator *aa, vector *x)
{
	OK_CHECK_PTR(aa);
	return OK_SCAN_ERR( anderson_accelerate_template(
		aa->blas_handle, aa->F, aa->G, aa->F_gram, aa->alpha,
		aa->ones, aa->mu_regularization, &aa->iter, x, aa->w,
		aa->n_blocks, anderson_sum_blocks) );
}

#ifdef __cplusplus
}
#endif

