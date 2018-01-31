#ifndef OPTKIT_POGS_DATATYPES_H_
#define OPTKIT_POGS_DATATYPES_H_

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
#ifndef OK_DEBUG_PYTHON
#define POGS_PRIVATE static
#undef OPTKIT_POGS_IMPLEMENTATION_
#else
#define POGS_PRIVATE
#endif
*/

#ifndef POGS_CONSTANTS
#define POGS_CONSTANTS
#define kALPHA (ok_float) 1.7
#define kMAXITER 2000u
#define kANDERSON 20u
#define kRHO (ok_float) 1
#define kATOL (ok_float) 1e-4
#define kRTOL (ok_float) 1e-3
#define kTOLPROJ (ok_float) 1e-6
#define kTOLADAPT (ok_float) 1
#define kTOLCORR (ok_float) 2e-1 /* TODO: keep? */
#define kMU (ok_float) 1e-3
#define kADAPTIVE 1
#define kADAPTSPECTRAL 0 /* TODO: keep? */
#define kACCELERATE 0 /* CHANGE THIS TO 1 AFTER DEBUGGING */
#define kGAPSTOP 0
#define kWARMSTART 0
#define kDIAGNOSTIC 0
#define kVERBOSE 2u
#define kSUPPRESS 0u
#define kRHOINTERVAL 2u /* TODO: keep? */
#define kRESUME 0
#endif /* POGS_CONSTANTS */

typedef struct POGSGraphVector {
	size_t size, m, n;
	vector *x, *y, *vec;
	int memory_attached;
} pogs_graph_vector;

typedef struct POGSResiduals {
	ok_float primal, dual, gap;
} pogs_residuals;

typedef struct POGSTolerances {
	ok_float primal, dual, gap;
	ok_float reltol, abstol, atolm, atoln, atolmn;
} pogs_tolerances;

typedef struct POGSObjectiveValues {
	ok_float primal, dual, gap;
} pogs_objective_values;

/* TODO: keep tolcorr, rho_interval, adaptspectral?? */
typedef struct POGSSettings {
	ok_float alpha, rho, abstol, reltol, tolproj, toladapt, tolcorr;
	ok_float anderson_regularization;
	uint maxiter, anderson_lookback, rho_interval, verbose, suppress;
	int adaptiverho, adapt_spectral, accelerate, gapstop, warmstart, resume, diagnostic;
	ok_float *x0, *nu0;
} pogs_settings;

typedef struct POGSInfo {
	int err;
	int converged;
	uint k;
	ok_float obj, rho, setup_time, solve_time;
} pogs_info;

typedef struct POGSOutput {
	ok_float *x, *y, *mu, *nu;
	ok_float *primal_residuals, *dual_residuals;
	ok_float *primal_tolerances, *dual_tolerances;
} pogs_output;

typedef struct POGSVariables {
	vector *state, *fixed_point_iterate;
	pogs_graph_vector *primal, *primal12, *dual, *dual12;
	pogs_graph_vector *prev, *temp;
	size_t m, n;
} pogs_variables;


typedef struct POGSConvergence {
	vector *primal, *dual, *primal_tol, *dual_tol;
} pogs_convergence;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPTKIT_POGS_DATATYPES_H_ */
