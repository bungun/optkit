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
// #define kRHOITERS (size_t) 2
// #define kEPSCORR (ok_float) 0.2
#endif /* SPECTRAL_RHO_CONSTANTS */

typedef struct SpectralRhoParameters {
	vector *dH, *dG, *dlambda, *dlambda12
	// size_t rho_iters;
	// ok_float eps_corr;
} spectral_rho_params;


ok_status pogs_spectral_rho_initialize(spectral_rho_params *params,
	const size_t dim)
{
	OK_CHECK_PTR(params);
	ok_status err = OPTKIT_SUCCESS;
	ok_alloc(&params->dH, sizeof(param->dH));
	ok_alloc(&params->dG, sizeof(param->dG));
	ok_alloc(&params->dlambda, sizeof(param->dlambda));
	ok_alloc(&params->dlambda12, sizeof(param->dlambda12));
	OK_CHECK_ERR(err, vector_calloc(params->dH, dim));
	OK_CHECK_ERR(err, vector_calloc(params->dG, dim));
	OK_CHECK_ERR(err, vector_calloc(params->dlambda, dim));
	OK_CHECK_ERR(err, vector_calloc(params->dlambda12, dim));
	// params->rho_iters = kRHOITERS;
	// params->eps_corr = kEPSCORR;
	if (err)
		pogs_spectral_rho_finish(params)
	return err;
}

ok_status pogs_spectral_rho_free(spectral_rho_params *params)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_MAX_ERR(err, vector_free(params->dH));
	OK_MAX_ERR(err, vector_free(params->dH));
	OK_MAX_ERR(err, vector_free(params->dlambda));
	OK_MAX_ERR(err, vector_free(params->dlambda12));
	ok_free(params->dH);
	ok_free(params->dG);
	ok_free(params->dlambda);
	ok_free(params->dlambda12);
	return err;
}

ok_status pogs_spectral_update_start(spectral_rho_params *params,
	const pogs_variables *z, const ok_float rho)
{
	if (!params || !z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR(err, vector_copy(params->dH, z->primal12->vec));
	OK_CHECK_ERR(err, vector_copy(params->dG, z->primal->vec));
	OK_CHECK_ERR(err, vector_scale(params->dG, -kOne));
	OK_CHECK_ERR(err, vector_copy(params->dlambda, z->dual->vec));
	OK_CHECK_ERR(err, vector_scale(params->dG, -rho));
	OK_CHECK_ERR(err, vector_copy(params->dlambda12, z->dual12->vec));
	OK_CHECK_ERR(err, vector_scale(params->dG, -rho));
	return err;
}

ok_status pogs_spectral_update_end(void *linalg_handle,
	spectral_rho_params *params, const pogs_variables *z,
	const ok_float rho)
{
	if (!params || !z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR(err, blas_axpy(-kOne, z->primal12->vec, params->dH));
	OK_CHECK_ERR(err, blas_axpy(kOne, z->primal->vec, params->dG));
	OK_CHECK_ERR(err, blas_axpy(rho, z->dual->vec, params->dlambda));
	OK_CHECK_ERR(err, blas_axpy(rho, z->dual12->vec, params->dlambda12));
	return err;
}

ok_status pogs_spectral_estimate_tangent(void *linalg_handle, vector *dF,
	vector *dx, ok_float *slope, ok_float *corr)
{
	OK_CHECK_PTR(slope);
	OK_CHECK_PTR(corr);
	ok_status err = OPTKIT_SUCCESS;
	ok_float dFdx, dFdF, dxdx, max_gradient, steepest_descent;
	ok_float Fnorm, xnorm;

	OK_CHECK_ERR(err, blas_dot(hdl, dx, dF, &dFdx));
	OK_CHECK_ERR(err, blas_dot(hdl, dx, dx, &dxdx));
	OK_CHECK_ERR(err, blas_dot(hdl, dF, dF, &dFdF));

	max_gradient = dFdx / dFdF;
	steepest_descent = dxdx / dFdx;

	if (steepest_descent < 0.5 * max_gradient)
		*slope = max_gradient;
	else
		*slope = steepest_descent - 0.5 * max_gradient;

	OK_CHECK_ERR(err, blas_nrm2(hdl, dx, &xnorm));
	OK_CHECK_ERR(err, blas_nrm2(hdl, dF, &Fnorm));
	*corr = dFdx / (Fnorm * xnorm);

	return err;
}

/*
 * change solver->rho to balance primal and dual convergence
 * (and rescale z->dual accordingly)
 */
ok_status pogs_spectral_adapt_rho(void *hdl, pogs_variables *z, ok_float *rho,
	spectral_params *params, const pogs_settings *settings, const uint k)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_float rho_new;
	ok_float alpha_corr, alpha, beta_corr, beta;

	if (!z || !rho || !params || !(params->dH) || !settings || !res || !tol)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (!(settings->adaptiverho) || (k % != settings->rho_interval))
		return OPTKIT_SUCCESS;

	/* complete differences from previous iteration */
	pogs_spectral_udate_end(hdl, params, z, rho);

	/* estimate tangents from differences */
	OK_CHECK_ERR(err, pogs_spectral_estimate_tangent(hdl, params->dH,
		params->dlambda12, &alpha, &alpha_corr));
	OK_CHECK_ERR(err, pogs_spectral_estimate_tangent(hdl, params->dG,
		params->dlambda, &beta, &beta_corr));

	/* safeguards */
	if (alpha_corr > settings->tolcorr && beta_corr > settings->tolcorr)
		rho_new = MATH(sqrt)(alpha * beta);
	else if (alpha_corr > settings->tolcorr)
		rho_new = alpha;
	else if (beta_corr > settings->tolcorr)
		rho_new = beta;
	else
		rho_new = rho;

	/* start building differences for next iteration */
	pogs_spectral_udate_start(params, z, rho_new);

	OK_CHECK_ERR(err, vector_scale(z->dual->vec, rho / rho_new));
	*rho = rho_new;

	return err;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPTKIT_POGS_SPECTRAL_RHO_H_ */

