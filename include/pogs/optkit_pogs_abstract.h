#ifndef OPTKIT_POGS_ABSTRACT_H_
#define OPTKIT_POGS_ABSTRACT_H_

#include "optkit_abstract_operator.h"
#include "optkit_operator_dense.h"
#include "optkit_operator_sparse.h"
#include "optkit_operator_typesafe.h"
#include "optkit_pogs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct POGSSolver {
	pogs_work * W;
	pogs_variables * z;
	FunctionVector * f, * g;
	ok_float rho;
	pogs_settings * settings;
	void * linalg_handle;
	ok_float init_time;
} pogs_solver;

POGS_PRIVATE ok_status pogs_work_alloc(pogs_work ** W, operator * A, int direct);
POGS_PRIVATE ok_status pogs_work_free(pogs_work * W);
POGS_PRIVATE ok_status pogs_solver_alloc(pogs_solver ** solver, operator * A,
	int direct);
POGS_PRIVATE ok_status pogs_solver_free(pogs_solver * solver);
POGS_PRIVATE ok_status equilibrate(void * linalg_handle, pogs_work * W,
	const ok_float pnorm);
POGS_PRIVATE ok_float estimate_norm(void * linalg_handle, pogs_work * W);
POGS_PRIVATE ok_status normalize_DAE(void * linalg_handle, pogs_work * W);
POGS_PRIVATE ok_status update_problem(pogs_solver * solver, FunctionVector * f,
	FunctionVector * g);
POGS_PRIVATE ok_status initialize_variables(pogs_solver * solver);
POGS_PRIVATE ok_status update_residuals(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps);
POGS_PRIVATE int check_convergence(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps,
	int gapstop);
POGS_PRIVATE ok_status project_primal(void * linalg_handle, projector * proj,
	pogs_variables * z, ok_float alpha, ok_float tol);
POGS_PRIVATE ok_status pogs_solver_loop(pogs_solver * solver, pogs_info * info);

pogs_solver * pogs_init(operator * A, const int direct,
	const ok_float equil_norm);
ok_status pogs_solve(pogs_solver * solver, FunctionVector * f,
	FunctionVector * g, const pogs_settings * settings, pogs_info * info,
	pogs_output * output);
ok_status pogs_finish(pogs_solver * solver, const int reset);
ok_status pogs(operator * A, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	const int direct, const ok_float equil_norm, const int reset);

operator * pogs_dense_operator_gen(const ok_float * A, size_t m, size_t n,
	enum CBLAS_ORDER order);
operator * pogs_sparse_operator_gen(const ok_float * val, const ok_int * ind,
	const ok_int * ptr, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order);
ok_status pogs_dense_operator_free(operator * A);
ok_status pogs_sparse_operator_free(operator * A);
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

#endif /* OPTKIT_POGS_ABSTRACT_H_ */
