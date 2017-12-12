#include "optkit_anderson_difference.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_status anderson_fused_diff_accelerator_init(fused_diff_accelerator *aa,
	size_t vector_dim, size_t lookback_dim, size_t n_blocks)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(aa);
	if (aa->DF)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	if (lookback_dim > vector_dim)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
		// lookback_dim = vector_dim;
	if (vector_dim % n_blocks != 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_LAYOUT_MISMATCH );

	aa->vector_dim = vector_dim;
	aa->lookback_dim = lookback_dim;
	aa->n_blocks = n_blocks;
	aa->iter = 0;

	ok_alloc(aa->DX, sizeof(*aa->DX));
	OK_CHECK_ERR( err, matrix_calloc(aa->DX, vector_dim / n_blocks,
		lookback_dim - 1, CblasColMajor) );
	ok_alloc(aa->DF, sizeof(*aa->DF));
	OK_CHECK_ERR( err, matrix_calloc(aa->DF, vector_dim / n_blocks,
		lookback_dim - 1, CblasColMajor) );
	ok_alloc(aa->DG, sizeof(*aa->DG));
	OK_CHECK_ERR( err, matrix_calloc(aa->DG, vector_dim, lookback_dim - 1,
		CblasColMajor) );
	ok_alloc(aa->DXDF, sizeof(*aa->DXDF));
	OK_CHECK_ERR( err, matrix_calloc(aa->DXDF, lookback_dim - 1,
		lookback_dim - 1, CblasColMajor) );

	ok_alloc(aa->w, sizeof(*aa->w));
	OK_CHECK_ERR( err, vector_calloc(aa->w, vector_dim / n_blocks) );
	ok_alloc(aa->f, sizeof(*aa->f));
	OK_CHECK_ERR( err, vector_calloc(aa->f, vector_dim / n_blocks) );
	ok_alloc(aa->g, sizeof(*aa->g));
	OK_CHECK_ERR( err, vector_calloc(aa->g, vector_dim) );
	ok_alloc(aa->x, sizeof(*aa->x));
	OK_CHECK_ERR( err, vector_calloc(aa->x, vector_dim / n_blocks) );
	ok_alloc(aa->alpha, sizeof(*aa->alpha));
	OK_CHECK_ERR( err, vector_calloc(aa->alpha, lookback_dim - 1) );

	ok_alloc(aa->pivot, sizeof(*aa->pivot));
	OK_CHECK_ERR( err, int_vector_calloc(aa->pivot, lookback_dim - 1) );

	OK_CHECK_ERR( err, blas_make_handle(&(aa->blas_handle)) );
	OK_CHECK_ERR( err, lapack_make_handle(&(aa->lapack_handle)) );

	if (err)
		OK_MAX_ERR( err, anderson_fused_diff_accelerator_free(aa) );

	return err;
}

ok_status anderson_fused_diff_accelerator_free(fused_diff_accelerator *aa)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(aa);
	OK_MAX_ERR( err, matrix_free(aa->DX) );
	OK_MAX_ERR( err, matrix_free(aa->DF) );
	OK_MAX_ERR( err, matrix_free(aa->DG) );
	OK_MAX_ERR( err, matrix_free(aa->DXDF) );

	OK_MAX_ERR( err, vector_free(aa->w) );
	OK_MAX_ERR( err, vector_free(aa->f) );
	OK_MAX_ERR( err, vector_free(aa->g) );
	OK_MAX_ERR( err, vector_free(aa->x) );
	OK_MAX_ERR( err, vector_free(aa->alpha) );

	OK_MAX_ERR( err, int_vector_free(aa->pivot) );

	OK_MAX_ERR( err, blas_destroy_handle(aa->blas_handle) );
	OK_MAX_ERR( err, blas_destroy_handle(aa->lapack_handle) );

	ok_free(aa->DX);
	ok_free(aa->DF);
	ok_free(aa->DG);
	ok_free(aa->DXDF);

	ok_free(aa->w);
	ok_free(aa->f);
	ok_free(aa->g);
	ok_free(aa->x);
	ok_free(aa->alpha);
	ok_free(aa->pivot);
	return err;
}


ok_status anderson_fused_diff_set_x0(fused_diff_accelerator *aa,
	vector *x_initial)
{
	ok_status err = OPTKIT_SUCCESS;
	vector xcol;
	OK_CHECK_VECTOR(x_initial);
	if (x_initial->size != aa->vector_dim)
		err = OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_CHECK_ERR( err, anderson_sum_blocks(aa->w, x_initial, aa->n_blocks) );
	OK_CHECK_ERR( err, matrix_column(&xcol, aa->DX, 0) );
	OK_CHECK_ERR( err, vector_sub(&xcol, aa->w) );
	OK_CHECK_ERR( err, vector_memcpy_vv(aa->x, aa->w) );
	return err;
}

ok_status anderson_fused_diff_accelerate(fused_diff_accelerator *aa,
	vector *x)
{
	OK_CHECK_PTR(aa);
	return OK_SCAN_ERR( anderson_difference_accelerate_template(
		aa->blas_handle, aa->lapack_handle, aa->DX, aa->DF, aa->DG,
		aa->DXDF, aa->f, aa->g, aa->x, aa->alpha, aa->pivot, &aa->iter,
		x, aa->w, aa->n_blocks, anderson_sum_blocks) );

}

#ifdef __cplusplus
}
#endif

