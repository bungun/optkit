#ifndef OPTKIT_POGS_HYBRID_RHO_H
#define OPTKIT_POGS_HYBRID_RHO_H

#include "optkit_pogs_adaptive_rho.h"
#include "optkit_pogs_spectral_rho.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HybridRhoParameters {
	adapt_params *adapt_params;
	spectral_rho_params *spectral_params;
	ok_float breakpoint;
	int spectral;
} rho_params;

ok_status pogs_hybrid_rho_initialize(rho_params *params,
	const pogs_settings *settings, const pogs_variables *z)
{
	ok_status err = OPTKIT_SUCCESS;

	if (!params || !settings)
		return OK_SCAN_ERR(OPTKIT_ERROR_UNALLOCATED);

	params->breakpoint = settings->rho_breakpoint;
	if (settings->rho_breakpoint <= 0)
		params->spectral = 0;
	else
		params->spectral = settings->adapt_spectral;

	ok_alloc(params->adapt_params, sizeof(*params->adapt_params));
	if (params->spectral) {
		ok_alloc(params->spectral_params, sizeof(*params->spectral_params));
		OK_CHECK_ERR(err, pogs_spectral_rho_initialize(
			params->spectral_params, z->m + z->n));
		OK_CHECK_ERR(err, pogs_spectral_update_start(
			params->spectral_params, z, settings->rho));
	}
	return err;
}

ok_status pogs_hybrid_rho_free(rho_params *params)
{
	ok_status err = OPTKIT_SUCCESS;
	if (params->spectral_params)
		OK_CHECK_ERR(err, pogs_spectral_rho_free(params->spectral_params));
		ok_free(params->spectral_params);
	ok_free(params->adapt_params);
	return err;
}

ok_status pogs_hybrid_adapt_rho(void *blas_handle, pogs_variables *z,
	ok_float *rho, rho_params *params, const pogs_settings *settings,
	const pogs_residuals *res, const pogs_tolerances *tol, const uint k)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_float tol_scal;
	int primal_nogo, dual_nogo;

	if (!params || !res || !tol)
		return OK_SCAN_ERR(OPTKIT_ERROR_UNALLOCATED);

	if (params->spectral) {
		/* test whether to break to standard adaptive rho heuristic */
		tol_scal = (params->breakpoint / tol->reltol);
		primal_nogo = res->primal > tol_scal * tol->primal;
		dual_nogo = res->dual > tol_scal * tol->dual;
		params->spectral = primal_nogo * dual_nogo;
		if (!params->spectral)
			printf("%s\n", "SWITCHING FROM SPECTRAL TO ADAPTIVE");
			/* TODO: remove print statement */
	}

	if (params->spectral)
		OK_CHECK_ERR(err, pogs_spectral_adapt_rho(
			blas_handle, z, rho, params->spectral_params,
			settings, k));
	else
		OK_CHECK_ERR(err, pogs_adapt_rho(
			z, rho, params->adapt_params, settings, res, tol, k));

	return err;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPTKIT_POGS_HYBRID_RHO_H */
