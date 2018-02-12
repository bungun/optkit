/*
 * Implement adaptive rho method proposed in https://arxiv.org/pdf/1605.07246.pdf
 *
 */
#ifndef OPTKIT_POGS_SPECTRAL_RHO_H_
#define OPTKIT_POGS_SPECTRAL_RHO_H_

#include "optkit_pogs_datatypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SPECTRAL_RHO_CONSTANTS
#define SPECTRAL_RHO_CONSTANTS
#define kRHOMIN_SPECTRAL (ok_float) 1e-5
#define kRHOMAX_SPECTRAL (ok_float) 1e5
// #define kRHOITERS (size_t) 2
// #define kEPSCORR (ok_float) 0.2
#endif /* SPECTRAL_RHO_CONSTANTS */

typedef struct SpectralRhoParameters {
	vector *dH, *dG, *dlambda, *dlambda12;
	// size_t rho_iters;
	// ok_float eps_corr;
} spectral_rho_params;


ok_status pogs_spectral_rho_free(spectral_rho_params *params)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_MAX_ERR(err, vector_free(params->dH));
	OK_MAX_ERR(err, vector_free(params->dG));
	OK_MAX_ERR(err, vector_free(params->dlambda));
	OK_MAX_ERR(err, vector_free(params->dlambda12));
	ok_free(params->dH);
	ok_free(params->dG);
	ok_free(params->dlambda);
	ok_free(params->dlambda12);
	return err;
}

ok_status pogs_spectral_rho_initialize(spectral_rho_params *params,
	const size_t dim)
{
	OK_CHECK_PTR(params);
	ok_status err = OPTKIT_SUCCESS;
	ok_alloc(params->dH, sizeof(*params->dH));
	ok_alloc(params->dG, sizeof(*params->dG));
	ok_alloc(params->dlambda, sizeof(*params->dlambda));
	ok_alloc(params->dlambda12, sizeof(*params->dlambda12));
	OK_CHECK_ERR(err, vector_calloc(params->dH, dim));
	OK_CHECK_ERR(err, vector_calloc(params->dG, dim));
	OK_CHECK_ERR(err, vector_calloc(params->dlambda, dim));
	OK_CHECK_ERR(err, vector_calloc(params->dlambda12, dim));
	// params->rho_iters = kRHOITERS;
	// params->eps_corr = kEPSCORR;
	if (err)
		pogs_spectral_rho_free(params);
	return err;
}

ok_status pogs_spectral_update_start(spectral_rho_params *params,
	const pogs_variables *z, const ok_float rho)
{
	if (!params || !z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR(err, vector_memcpy_vv(params->dH, z->primal12->vec));
	OK_CHECK_ERR(err, vector_memcpy_vv(params->dG, z->primal->vec));
	OK_CHECK_ERR(err, vector_scale(params->dG, -kOne));
	OK_CHECK_ERR(err, vector_memcpy_vv(params->dlambda, z->dual->vec));
	OK_CHECK_ERR(err, vector_scale(params->dlambda, -rho));
	OK_CHECK_ERR(err, vector_memcpy_vv(params->dlambda12, z->dual12->vec));
	OK_CHECK_ERR(err, vector_scale(params->dlambda12, -rho));
	return err;
}

ok_status pogs_spectral_update_end(void *blas_handle,
	spectral_rho_params *params, const pogs_variables *z,
	const ok_float rho)
{
	if (!params || !z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR(err, blas_axpy(blas_handle, -kOne, z->primal12->vec, params->dH));
	OK_CHECK_ERR(err, blas_axpy(blas_handle, kOne, z->primal->vec, params->dG));
	OK_CHECK_ERR(err, blas_axpy(blas_handle, rho, z->dual->vec, params->dlambda));
	OK_CHECK_ERR(err, blas_axpy(blas_handle,rho, z->dual12->vec, params->dlambda12));
	return err;
}

ok_status pogs_spectral_estimate_tangent(void *blas_handle, vector *dF,
	vector *dx, ok_float *slope, ok_float *corr)
{
	OK_CHECK_PTR(slope);
	OK_CHECK_PTR(corr);
	ok_status err = OPTKIT_SUCCESS;
	ok_float dFdx = kOne, dFdF = kOne, dxdx = kZero;
	ok_float min_gradient = kZero, steepest_descent = kZero;
	ok_float Fnorm = kOne, xnorm = kOne;

	OK_CHECK_ERR(err, blas_dot(blas_handle, dx, dF, &dFdx));
	OK_CHECK_ERR(err, blas_dot(blas_handle, dx, dx, &dxdx));
	OK_CHECK_ERR(err, blas_dot(blas_handle, dF, dF, &dFdF));

	min_gradient = dFdx / dFdF;
	steepest_descent = dxdx / dFdx;

	if (steepest_descent < (ok_float) 0.5 * min_gradient)
		*slope = min_gradient;
	else
		*slope = steepest_descent - (ok_float) 0.5 * min_gradient;

	OK_CHECK_ERR(err, blas_nrm2(blas_handle, dx, &xnorm));
	OK_CHECK_ERR(err, blas_nrm2(blas_handle, dF, &Fnorm));
	*corr = dFdx / (Fnorm * xnorm);

	return err;
}

/*
 * change solver->rho to balance primal and dual convergence
 * (and rescale z->dual accordingly)
 */
ok_status pogs_spectral_adapt_rho(void *blas_handle, pogs_variables *z,
	ok_float *rho, spectral_rho_params *params,
	const pogs_settings *settings, const uint k)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_float rho_new;
	ok_float alpha_corr = kZero, alpha, beta_corr = kZero, beta;

	if (!z || !rho || !params || !(params->dH) || !settings)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (!(settings->adaptiverho) || (k % settings->rho_interval != 0))
		return OPTKIT_SUCCESS;

	/* complete differences from previous iteration */
	OK_CHECK_ERR(err, pogs_spectral_update_end(blas_handle, params, z,
		*rho));

	/* estimate tangents from differences */
	OK_CHECK_ERR(err, pogs_spectral_estimate_tangent(blas_handle,
		params->dH, params->dlambda12, &alpha, &alpha_corr));
	OK_CHECK_ERR(err, pogs_spectral_estimate_tangent(blas_handle,
		params->dG, params->dlambda, &beta, &beta_corr));

	/* safeguards */
	if (alpha_corr > settings->tolcorr && beta_corr > settings->tolcorr)
		rho_new = MATH(sqrt)(alpha * beta);
	else if (alpha_corr > settings->tolcorr)
		rho_new = alpha;
	else if (beta_corr > settings->tolcorr)
		rho_new = beta;
	else
		rho_new = *rho;

	/* fallback */
	if (err)
		rho_new = *rho;

	rho_new = (rho_new < kRHOMIN_SPECTRAL) ? kRHOMIN_SPECTRAL : rho_new;
	rho_new = (rho_new > kRHOMAX_SPECTRAL) ? kRHOMAX_SPECTRAL : rho_new;

	OK_CHECK_ERR(err, blas_scal(blas_handle, *rho / rho_new, z->dual->vec));
	*rho = rho_new;

	/* start building differences for next iteration */
	OK_CHECK_ERR(err, pogs_spectral_update_start(params, z, *rho));

	return err;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPTKIT_POGS_SPECTRAL_RHO_H_ */
