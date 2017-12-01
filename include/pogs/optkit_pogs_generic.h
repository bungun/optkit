#ifndef OPTKIT_POGS_H_
#define OPTKIT_POGS_H_

#include "optkit_timer.h"
#include "optkit_prox.hpp"
#include "optkit_anderson.h"
#include "optkit_pogs_datatypes.h"
#include "optkit_pogs_adaptive_rho.h"
#include "optkit_pogs_impl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	OkPogsDense = 1001,
	OkPogsSparse = 1002,
	OkPogsAbstract = 2001
} ok_pogs_impl;

#undef OK_POGS_TYPE
/* SWITCH BETWEEN POGS IMPLEMENTATIONS */

#ifdef OK_COMPILE_POGS_ABSTRACT
	#define OK_POGS_TYPE OkPogsAbstract
	#undef OK_COMPILE_POGS_SPARSE
	#undef OK_COMPILE_POGS_DENSE
	#include "optkit_pogs_impl_abstract.h"
	typedef pogs_abstract_work pogs_work;
	typedef abstract_operator pogs_solver_data;
	typedef pogs_abstract_solver_flags pogs_solver_flags;
	typedef pogs_abstract_solver_private_data pogs_solver_private_data;
	#define POGS(x) pogs_abstract_ ## x
#endif

/*
#ifdef OK_COMPILE_POGS_SPARSE
	#define OK_POGS_TYPE OkPogsSparse
	#undef OK_COMPILE_POGS_ABSTRACT
	#undef OK_COMPILE_POGS_DENSE
	#include "optkit_pogs_impl_sparse.h"
	typedef pogs_sparse_matrix pogs_work;
	typedef pogs_sparse_solver_data pogs_solver_data;
	typedef pogs_sparse_solver_flags pogs_solver_flags;
	typedef pogs_sparse_solver_private_data pogs_solver_private_data;
	#define POGS(x) pogs_sparse_ ## x
#endif
*/

#ifndef OK_POGS_TYPE
	#define OK_POGS_TYPE OkPogsDense
	#undef OK_COMPILE_POGS_ABSTRACT
	#undef OK_COMPILE_POGS_SPARSE
	#include "optkit_pogs_impl_dense.h"
	typedef pogs_dense_work pogs_work;
	typedef ok_float pogs_solver_data;
	typedef pogs_dense_solver_flags pogs_solver_flags;
	typedef pogs_dense_solver_private_data pogs_solver_private_data;
	#define POGS(x) pogs_dense_ ## x
#endif

typedef struct POGSSolver {
	pogs_work * W;
	pogs_variables * z;
	function_vector * f, * g;
	ok_float rho;
	pogs_settings * settings;
	void * linalg_handle;
	ok_float init_time;
	anderson_accelerator * aa;
	pogs_convergence * convergence;
} pogs_solver;

ok_pogs_impl get_pogs_impl(void);

ok_status pogs_work_alloc(pogs_work * W, pogs_solver_data * A,
	const pogs_solver_flags * flags);
ok_status pogs_work_free(pogs_work * W);

ok_status pogs_solver_alloc(pogs_solver * solver, pogs_solver_data * A,
	const pogs_solver_flags * flags);
ok_status pogs_solver_free(pogs_solver * solver);

ok_status pogs_normalize_DAE(pogs_work * W);
ok_status pogs_set_z0(pogs_solver * solver);

ok_status pogs_primal_update(pogs_variables * z);
ok_status pogs_prox(void * linalg_handle, function_vector * f,
	function_vector * g, pogs_variables * z, ok_float rho);
ok_status pogs_project_graph(pogs_work * W, pogs_variables * z, ok_float alpha,
	ok_float tol);
ok_status pogs_dual_update(void * linalg_handle, pogs_variables * z,
	ok_float alpha);

ok_status pogs_iterate(pogs_solver * solver);
ok_status pogs_accelerate(pogs_solver * solver);
ok_status pogs_update_residuals(pogs_solver * solver,
	pogs_objective_values * obj, pogs_residuals * res);
ok_status pogs_check_convergence(pogs_solver * solver,
	pogs_objective_values * obj, pogs_residuals * res,
	pogs_tolerances * tol, int * converged);

ok_status pogs_setup_diagnostics(pogs_solver * solver, const uint iters);
ok_status pogs_record_diagnostics(pogs_solver * solver,
	const pogs_residuals * res, const uint iter);
ok_status pogs_emit_diagnostics(pogs_output * output, pogs_solver * solver);

ok_status pogs_solver_loop(pogs_solver * solver, pogs_info * info);

pogs_solver * pogs_init(pogs_solver_data * A, const pogs_solver_flags * flags);
ok_status pogs_solve(pogs_solver * solver, const function_vector * f,
	const function_vector * g, const pogs_settings * settings,
	pogs_info * info, pogs_output * output);
ok_status pogs_finish(pogs_solver * solver, const int reset);
ok_status pogs(pogs_solver_data * A, const pogs_solver_flags * flags,
	const function_vector * f, const function_vector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	const int reset);

ok_status pogs_export_solver(pogs_solver_private_data * data, ok_float * state,
	ok_float * rho, pogs_solver_flags * flags, const pogs_solver * solver);
pogs_solver * pogs_load_solver(const pogs_solver_private_data * data,
	const ok_float * state, const ok_float rho,
	const pogs_solver_flags * flags);

ok_status pogs_solver_save_state(ok_float * state, ok_float * rho,
	const pogs_solver * solver);
ok_status pogs_solver_load_state(pogs_solver * solver, const ok_float * state,
	const ok_float rho);


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_POGS_H_ */

