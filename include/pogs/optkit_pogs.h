#ifndef OPTKIT_POGS_H_
#define OPTKIT_POGS_H_

#include "optkit_timer.h"
#include "optkit_equilibration.h"
#include "optkit_projector.h"

#ifndef OPTKIT_POGS_IMPLEMENTATION_
#define OPTKIT_POGS_IMPLEMENTATION_
#endif

#include "optkit_pogs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OPTKIT_INDIRECT
typedef direct_projector projector_;
#define PROJECTOR(x) direct_projector_ ## x
#else
typedef indirect_projector projector_;
#define PROJECTOR(x) indirect_projector_ ## x
#endif

typedef struct POGSMatrix {
	matrix * A;
	projector_ * P;
	vector * d, * e;
	ok_float normA;
	int skinny, normalized, equilibrated;
} pogs_matrix;

typedef struct POGSSolver {
	pogs_matrix * M;
	pogs_variables * z;
	function_vector * f, * g;
	ok_float rho;
	pogs_settings * settings;
	void * linalg_handle;
	ok_float init_time;
} pogs_solver;

int is_direct(void);

POGS_PRIVATE ok_status pogs_matrix_alloc(pogs_matrix ** M, size_t m, size_t n,
	enum CBLAS_ORDER ord);
POGS_PRIVATE ok_status pogs_matrix_free(pogs_matrix * M);
POGS_PRIVATE ok_status pogs_solver_alloc(pogs_solver ** solver, size_t m,
	size_t n, enum CBLAS_ORDER ord);
POGS_PRIVATE ok_status pogs_solver_free(pogs_solver * solver);
POGS_PRIVATE ok_status equilibrate(void * linalg_handle, ok_float * A_orig,
	pogs_matrix * M, enum CBLAS_ORDER ord);
POGS_PRIVATE ok_status estimate_norm(void * linalg_handle, pogs_matrix * M,
	ok_float * normest);
POGS_PRIVATE ok_status normalize_DAE(void * linalg_handle, pogs_matrix * M);
POGS_PRIVATE ok_status update_problem(pogs_solver * solver, function_vector * f,
	function_vector * g);
POGS_PRIVATE ok_status initialize_variables(pogs_solver * solver);
POGS_PRIVATE pogs_tolerances make_tolerances(const pogs_settings * settings,
	size_t m, size_t n);
POGS_PRIVATE ok_status update_residuals(void * linalg_handle,
	pogs_solver * solver, pogs_objectives * obj, pogs_residuals * res,
	pogs_tolerances * eps);
POGS_PRIVATE int check_convergence(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps);
POGS_PRIVATE ok_status project_primal(void * linalg_handle, projector_ * proj,
	pogs_variables * z,  ok_float alpha);
POGS_PRIVATE ok_status pogs_solver_loop(pogs_solver * solver, pogs_info * info);

pogs_solver * pogs_init(ok_float * A, size_t m, size_t n, enum CBLAS_ORDER ord);
ok_status pogs_solve(pogs_solver * solver, function_vector * f,
	function_vector * g, const pogs_settings * settings, pogs_info * info,
	pogs_output * output);
ok_status pogs_finish(pogs_solver * solver, int reset);
ok_status pogs(ok_float * A, function_vector * f, function_vector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	enum CBLAS_ORDER ord, int reset);
pogs_solver * pogs_load_solver(ok_float * A_equil,
	ok_float * LLT_factorization, ok_float * d,
	ok_float * e, ok_float * z, ok_float * z12, ok_float * z_dual,
	ok_float * z_dual12, ok_float * z_prev, ok_float rho,
	size_t m, size_t n, enum CBLAS_ORDER ord);
ok_status pogs_extract_solver(pogs_solver * solver, ok_float * A_equil,
	ok_float * LLT_factorization, ok_float * d,
	ok_float * e, ok_float * z, ok_float * z12, ok_float * z_dual,
	ok_float * z_dual12, ok_float * z_prev, ok_float * rho,
	enum CBLAS_ORDER ord);
// ok_status pogs_load_state(pogs_solver * solver, const ok_float * z,
// 	const ok_float * z12, const ok_float * z_dual,
// 	const ok_float * z_dual12, const ok_float * z_prev,
// 	const ok_float * rho);
// { _pogs_load_state(solver->z, solver->settings, z, z12, z_dual, z_dual12, z_prev, rho);}

// ok_status pogs_extract_state(const pogs_solver * solver, ok_float * z,
// 	ok_float * z12, ok_float * z_dual, ok_float * z_dual12,
// 	ok_float * z_prev, ok _float * rho);
// { _pogs_extract_state_state(solver->z, solver->settings, z, z12, z_dual, z_dual12, z_prev, rho);}
ok_status pogs_solver_exists(pogs_solver * solver);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_POGS_H_ */
