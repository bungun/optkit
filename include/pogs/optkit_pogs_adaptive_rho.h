#ifndef OPTKIT_POGS_ADAPTIVE_RHO_H_
#define OPTKIT_POGS_ADAPTIVE_RHO_H_

#include "optkit_pogs_datatypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ADAPTIVE_RHO_CONSTANTS
#define ADAPTIVE_RHO_CONSTANTS
#define kRHOMAX (ok_float) 1e4
#define kRHOMIN (ok_float) 1e-4
#define kDELTAMAX (ok_float) 2.
#define kDELTAMIN (ok_float) 1.05
#define kGAMMA (ok_float) 1.01
#define kKAPPA (ok_float) 0.9
#define kTAU (ok_float) 0.8
#endif /* ADAPTIVE_RHO_CONSTANTS */

typedef struct AdaptiveRhoParameters {
	ok_float delta, l, u, xi;
} adapt_params;


ok_status pogs_adaptive_rho_initialize(adapt_params *params)
{
	OK_CHECK_PTR(params);
	params->delta = kDELTAMIN;
	params->l = kZero;
	params->u = kZero;
	params->xi = kOne;
	return OPTKIT_SUCCESS;
}

/*
 * change solver->rho to balance primal and dual convergence
 * (and rescale z->dual accordingly)
 */
ok_status pogs_adapt_rho(pogs_variables *z, ok_float *rho, adapt_params *params,
	const pogs_settings *settings, const pogs_residuals *res,
	const pogs_tolerances *tol, const uint k)
{
	if (!(settings->adaptiverho))
		return OPTKIT_SUCCESS;

	if (!z || !rho || !params || !res || !tol)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	if (res->dual < params->xi * settings->toladapt * tol->dual &&
		res->primal > params->xi * settings->toladapt * tol->primal &&
		kTAU * (ok_float) k > params->l) {

		if (*rho < kRHOMAX) {
			*rho *= params->delta;
			if (settings->verbose > 2)
				printf("+RHO: %.3e\n", *rho);
			vector_scale(z->dual->vec, kOne / params->delta);
			params->delta = (params->delta * kGAMMA < kDELTAMAX) ?
				params->delta * kGAMMA : kDELTAMAX;
			params->u = (ok_float) k;
		}
	} else if (res->dual > params->xi * settings->toladapt * tol->dual &&
			res->primal < params->xi * settings->toladapt * tol->primal &&
			kTAU * (ok_float) k > (ok_float) params->u) {

		if (*rho > kRHOMIN) {
			*rho /= params->delta;
			if (settings->verbose > 2)
				printf("-RHO: %.3e\n", *rho);

			vector_scale(z->dual->vec, params->delta);
			params->delta = (params->delta * kGAMMA < kDELTAMAX) ?
				  params->delta * kGAMMA : kDELTAMAX;
			params->l = (ok_float) k;
		}
	} else if (res->dual < params->xi * settings->toladapt * tol->dual &&
			res->primal < params->xi * settings->toladapt * tol->primal) {
		params->xi *= kKAPPA;
	} else {
		params->delta = (params->delta / kGAMMA > kDELTAMIN) ?
				params->delta / kGAMMA : kDELTAMIN;
	}
	return OPTKIT_SUCCESS;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPTKIT_POGS_ADAPTIVE_RHO_H_ */


