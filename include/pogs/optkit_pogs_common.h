#ifndef OPTKIT_POGS_COMMON_H_
#define OPTKIT_POGS_COMMON_H_

#include "optkit_dense.h"
#include "optkit_prox.hpp"
#include "optkit_equilibration.h"
#include "optkit_projector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OK_DEBUG_PYTHON
#define POGS_PRIVATE
#else
#define POGS_PRIVATE static
#endif

#ifndef POGS_CONSTANTS
#define POGS_CONSTANTS
#define kALPHA (ok_float) 1.7
#define kMAXITER 2000u
#define kRHO (ok_float) 1
#define kATOL (ok_float) 1e-4
#define kRTOL (ok_float) 1e-3
#define kADAPTIVE 1
#define kGAPSTOP 0
#define kWARMSTART 0
#define kVERBOSE 2u
#define kSUPPRESS 0u
#define kRESUME 0
#define kRHOMAX (ok_float) 1e4
#define kRHOMIN (ok_float) 1e-4
#define kDELTAMAX (ok_float) 2.
#define kDELTAMIN (ok_float) 1.05
#define kGAMMA (ok_float) 1.01
#define kKAPPA (ok_float) 0.9
#define kTAU (ok_float) 0.8
#endif /* POGS_CONSTANTS */

typedef struct AdaptiveRhoParameters {
	ok_float delta, l, u, xi;
} adapt_params;

typedef struct POGSBlockVector {
	size_t size, m, n;
	vector * x, * y, * vec;
} block_vector;

typedef struct POGSResiduals {
	ok_float primal, dual, gap;
} pogs_residuals;

typedef struct POGSTolerances {
	ok_float primal, dual, gap;
	ok_float reltol, abstol, atolm, atoln, atolmn;
} pogs_tolerances;

typedef struct POGSObjectives {
	ok_float primal, dual, gap;
} pogs_objectives;

typedef struct POGSSettings {
	ok_float alpha, rho, abstol, reltol;
	uint maxiter, verbose, suppress;
	int adaptiverho, gapstop, warmstart, resume;
	ok_float * x0, * nu0;
} pogs_settings;

typedef struct POGSInfo {
	int err;
	int converged;
	uint k;
	ok_float obj, rho, setup_time, solve_time;
} pogs_info;

typedef struct POGSOutput {
	ok_float * x, * y, * mu, * nu;
} pogs_output;

typedef struct POGSVariables {
	block_vector * primal, * primal12, * dual, * dual12;
	block_vector * prev, * temp;
	size_t m, n;
} pogs_variables;

int private_api_accessible(void);

POGS_PRIVATE ok_status block_vector_alloc(block_vector ** z, size_t m,
	size_t n);
POGS_PRIVATE ok_status block_vector_free(block_vector * z);
POGS_PRIVATE ok_status pogs_variables_alloc(pogs_variables ** z, size_t m,
	size_t n);
POGS_PRIVATE ok_status pogs_variables_free(pogs_variables * z);
POGS_PRIVATE ok_status update_settings(pogs_settings * settings,
	const pogs_settings * input);
POGS_PRIVATE ok_status initialize_conditions(pogs_objectives * obj,
	pogs_residuals * res, pogs_tolerances * tol,
	const pogs_settings * settings, size_t m, size_t n);
POGS_PRIVATE ok_status update_objective(void * linalg_handle,
	const function_vector * f, const function_vector * g, ok_float rho,
	pogs_variables * z, pogs_objectives * obj);
POGS_PRIVATE ok_status update_tolerances(void * linalg_handle,
	pogs_variables * z, pogs_objectives * obj, pogs_tolerances * eps);
POGS_PRIVATE ok_status set_prev(pogs_variables * z);
POGS_PRIVATE ok_status prox(void * linalg_handle, const function_vector * f,
	const function_vector * g, pogs_variables * z, ok_float rho);
POGS_PRIVATE ok_status update_dual(void * linalg_handle, pogs_variables * z,
	ok_float alpha);
POGS_PRIVATE ok_status adaptrho(pogs_variables * z,
	const pogs_settings * settings, ok_float * rho, adapt_params * params,
	const pogs_residuals * res, const pogs_tolerances * eps, const uint k);
POGS_PRIVATE ok_status copy_output(pogs_variables * z, vector * d, vector * e,
	pogs_output * output, const ok_float rho, const uint suppress);
POGS_PRIVATE ok_status print_header_string();
POGS_PRIVATE ok_status print_iter_string(pogs_residuals * res,
	pogs_tolerances * eps, pogs_objectives * obj, uint k);

ok_status set_default_settings(pogs_settings * settings);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_POGS_COMMON_H_ */
