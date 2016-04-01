#ifndef OPTKIT_ABSPOGS_H_
#define OPTKIT_ABSPOGS_H_

#include "optkit_abstract_operator.h"
#include "optkit_operator_dense.h"
#include "optkit_operator_sparse.h"
#include "optkit_operator_typesafe.h"
#include "optkit_prox.hpp"
#include "optkit_equilibration.h"
#include "optkit_projector.h"
#include "optkit_timer.h"

#ifdef __cplusplus
extern "C" {
#endif

void pogslib_version(int * maj, int * min, int * change, int * status);

#ifndef OK_DEBUG_PYTHON
#define POGS_PRIVATE
#else
#define POGS_PRIVATE static
#endif

const ok_float kALPHA = (ok_float) 1.7;
const uint kMAXITER = 2000u;
const ok_float kRHO = (ok_float) 1;
const ok_float kATOL = (ok_float) 1e-4;
const ok_float kRTOL = (ok_float) 1e-3;
const int kADAPTIVE = 1;
const int kGAPSTOP = 0;
const int kWARMSTART = 0;
const uint kVERBOSE = 2u;
const uint kSUPPRESS = 0u;
const int kRESUME = 0;
const ok_float kRHOMAX = (ok_float) 1e4;
const ok_float kRHOMIN = (ok_float) 1e-4;
const ok_float kDELTAMAX = (ok_float) 2.;
const ok_float kDELTAMIN = (ok_float) 1.05;
const ok_float kGAMMA = (ok_float) 1.01;
const ok_float kKAPPA = (ok_float) 0.9;
const ok_float kTAU = (ok_float) 0.8;

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

typedef struct POGSWork {
	operator * A;
	projector * P;
	vector * d, * e;
	ok_status (* operator_scale)(operator * o, const ok_float scaling);
	ok_status (* operator_equilibrate)(void * linalg_handle, operator * o,
		vector * d, vector * e, const ok_float pnorm);
	ok_float normA;
	int skinny, normalized, equilibrated;
} pogs_work;

typedef struct POGSVariables {
	block_vector * primal, * primal12, * dual, * dual12;
	block_vector * prev, * temp;
	size_t m, n;
} pogs_variables;

typedef struct POGSSolver {
	pogs_work * W;
	pogs_variables * z;
	FunctionVector * f, * g;
	ok_float rho;
	pogs_settings * settings;
	void * linalg_handle;
	ok_float init_time;
} pogs_solver;

POGS_PRIVATE void pogs_work_alloc(pogs_work ** W, operator * A, int direct);
POGS_PRIVATE void pogs_work_free(pogs_work * W);
POGS_PRIVATE void block_vector_alloc(block_vector ** z, size_t m, size_t n);
POGS_PRIVATE void block_vector_free(block_vector * z);
POGS_PRIVATE void pogs_variables_alloc(pogs_variables ** z, size_t m, size_t n);
POGS_PRIVATE void pogs_variables_free(pogs_variables * z);
POGS_PRIVATE void pogs_solver_alloc(pogs_solver ** solver, operator * A,
	int direct);
POGS_PRIVATE void pogs_solver_free(pogs_solver * solver);
POGS_PRIVATE void update_settings(pogs_settings * settings,
	const pogs_settings * input);
POGS_PRIVATE ok_status equilibrate(void * linalg_handle, pogs_work * W,
	const ok_float pnorm);
POGS_PRIVATE ok_float estimate_norm(void * linalg_handle, pogs_work * W);
POGS_PRIVATE ok_status normalize_DAE(void * linalg_handle, pogs_work * W);
POGS_PRIVATE void update_problem(pogs_solver * solver, FunctionVector * f,
	FunctionVector * g);
POGS_PRIVATE void initialize_variables(pogs_solver * solver);
POGS_PRIVATE pogs_tolerances make_tolerances(const pogs_settings * settings,
	size_t m, size_t n);
POGS_PRIVATE void update_objective(void * linalg_handle, FunctionVector * f,
	FunctionVector * g, ok_float rho, pogs_variables * z,
	pogs_objectives * obj);
POGS_PRIVATE void update_tolerances(void * linalg_handle, pogs_variables * z,
	pogs_objectives * obj, pogs_tolerances * eps);
POGS_PRIVATE void update_residuals(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps);
POGS_PRIVATE int check_convergence(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps,
	int gapstop);
POGS_PRIVATE void set_prev(pogs_variables * z);
POGS_PRIVATE void prox(void * linalg_handle, FunctionVector * f,
	FunctionVector * g, pogs_variables * z, ok_float rho);
POGS_PRIVATE void project_primal(void * linalg_handle, projector * proj,
	pogs_variables * z, ok_float alpha, ok_float tol);
POGS_PRIVATE void update_dual(void * linalg_handle, pogs_variables * z,
	ok_float alpha);
POGS_PRIVATE void adaptrho(pogs_solver * solver, adapt_params * params,
	pogs_residuals * res, pogs_tolerances * eps, uint k);
POGS_PRIVATE void copy_output(pogs_solver * solver, pogs_output * output);
POGS_PRIVATE void print_header_string();
POGS_PRIVATE void print_iter_string(pogs_residuals * res, pogs_tolerances * eps,
	pogs_objectives * obj, uint k);
POGS_PRIVATE void pogs_solver_loop(pogs_solver * solver, pogs_info * info);

/* DIRECT / INDIRECT IS A SETTING, DEFAULTS TO INDIRECT */
int private_api_accessible(void);
void set_default_settings(pogs_settings * settings);
pogs_solver * pogs_init(operator * A, int direct, ok_float equil_norm);
void pogs_solve(pogs_solver * solver, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output);
void pogs_finish(pogs_solver * solver, const int reset);
void pogs(operator * A, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	const int direct, const ok_float equil_norm, const int reset);

operator * pogs_dense_operator_gen(const ok_float * A, size_t m, size_t n,
	enum CBLAS_ORDER order);
operator * pogs_sparse_operator_gen(const ok_float * val, const ok_int * ind,
	const ok_int * ptr, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order);
void pogs_dense_operator_free(operator * A);
void pogs_sparse_operator_free(operator * A);
// pogs_solver * pogs_load_solver(operator * op_equil,
// 	operator * LLT_factorization, ok_float * d, ok_float * e, ok_float * z,
// 	ok_float * z12, ok_float * z_dual, ok_float * z_dual12,
// 	ok_float * z_prev, ok_float rho, size_t m, size_t n, int direct);
// void pogs_extract_solver(pogs_solver * solver, operator * op_equil,
// 	operator * LLT_factorization, ok_float * d, ok_float * e, ok_float * z,
// 	ok_float * z12, ok_float * z_dual, ok_float * z_dual12,
// 	ok_float * z_prev, ok_float * rho);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_ABSPOGS_H_ */
