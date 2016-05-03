#include "optkit_pogs_abstract.h"

#ifdef __cplusplus
extern "C" {
#endif

const ok_float kProjectorTolInitial = (ok_float) 1e-6;

POGS_PRIVATE ok_status pogs_work_alloc(pogs_work ** W, operator * A, int direct)
{
	ok_status err = OPTKIT_SUCCESS;

	if (*W)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	OK_CHECK_OPERATOR(A);

	int dense_or_sparse = (A->kind == OkOperatorDense ||
		A->kind == OkOperatorSparseCSC ||
		A->kind == OkOperatorSparseCSR);
	pogs_work * W_ = OK_NULL;
	ok_alloc(W_, sizeof(*W_));
	W_->A = A;

	/* set projector */
	if (direct && A->kind == OkOperatorDense)
		W_->P = dense_direct_projector_alloc(
				dense_operator_get_matrix_pointer(W_->A));
	else
		W_->P = indirect_projector_generic_alloc(W_->A);

	if (!W_->P)
		err = OPTKIT_ERROR_UNALLOCATED;

	/* set equilibration */
	if  (!dense_or_sparse) {
		W_->operator_equilibrate = operator_equilibrate;
		W_->operator_scale = typesafe_operator_scale;
	} else {
		W_->operator_equilibrate = operator_regularized_sinkhorn;
		if (A->kind == OkOperatorDense)
			W_->operator_scale = dense_operator_scale;
		else
			W_->operator_scale = sparse_operator_scale;
	}

	ok_alloc(W_->d, sizeof(*(W_->d)));
	ok_alloc(W_->e, sizeof(*(W_->e)));
	OK_CHECK_ERR( err, vector_calloc(W_->d, A->size1) );
	OK_CHECK_ERR( err, vector_calloc(W_->e, A->size2) );
	W_->skinny = (A->size1 >= A->size2);
	W_->normalized = 0;
	W_->equilibrated = 0;
	if (err)
		OK_MAX_ERR( err, pogs_work_free(W_) );
	else
		*W = W_;
	return err;
}

POGS_PRIVATE ok_status pogs_work_free(pogs_work * W)
{
	if (!W)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	ok_status err = OK_SCAN_ERR( W->P->free(W->P->data) );
	ok_free(W->P);
	OK_MAX_ERR( err, vector_free(W->d) );
	OK_MAX_ERR( err, vector_free(W->e) );
	ok_free(W);
	return err;
}

POGS_PRIVATE ok_status pogs_solver_alloc(pogs_solver ** solver, operator * A,
	int direct)
{
	ok_status err = OPTKIT_SUCCESS;
	pogs_solver * s = OK_NULL;

	if (*solver)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	OK_CHECK_OPERATOR(A);

	ok_alloc(s, sizeof(*s));
	ok_alloc(s->settings, sizeof(*(s->settings)));
	OK_CHECK_ERR(  err, set_default_settings(s->settings) );
	ok_alloc(s->f, sizeof(*(s->f)));
	ok_alloc(s->g, sizeof(*(s->g)));
	OK_CHECK_ERR( err, function_vector_calloc(s->f, A->size1) );
	OK_CHECK_ERR( err, function_vector_calloc(s->g, A->size2) );
	OK_CHECK_ERR( err, pogs_variables_alloc(&(s->z), A->size1, A->size2) );
	OK_CHECK_ERR( err, pogs_work_alloc(&(s->W), A, direct) );
	OK_CHECK_ERR( err, blas_make_handle(&(s->linalg_handle)) );
	s->rho = kOne;
	if (err)
		OK_MAX_ERR( err, pogs_solver_free(s) );
	else
		*solver = s;
	return err;
}

POGS_PRIVATE ok_status pogs_solver_free(pogs_solver * solver)
{
	OK_CHECK_PTR(solver);
	ok_status err = blas_destroy_handle(solver->linalg_handle);
	OK_MAX_ERR( err, pogs_work_free(solver->W) );
	OK_MAX_ERR( err, pogs_variables_free(solver->z) );
	ok_free(solver->settings);
	OK_MAX_ERR( err, function_vector_free(solver->f) );
	OK_MAX_ERR( err, function_vector_free(solver->g) );
	ok_free(solver->f);
	ok_free(solver->g);
	ok_free(solver);
	return err;
}

POGS_PRIVATE ok_status equilibrate(void * linalg_handle, pogs_work * W,
	const ok_float pnorm)
{
	ok_status err = W->operator_equilibrate(linalg_handle, W->A, W->d, W->e,
		pnorm);
	W->equilibrated = (err == OPTKIT_SUCCESS);
	return OK_SCAN_ERR( err );
}

POGS_PRIVATE ok_float estimate_norm(void * linalg_handle, pogs_work * W,
	ok_float * normest)
{
	return OK_SCAN_ERR(
		operator_estimate_norm(linalg_handle, W->A, normest) );
}

POGS_PRIVATE ok_status normalize_DAE(void * linalg_handle, pogs_work * W)
{
	OK_CHECK_PTR(W);

	ok_status err = OPTKIT_SUCCESS;
	size_t m = W->A->size1,  n = W->A->size2;
	size_t mindim = m < n ? m : n;
	ok_float sqrt_mindim = MATH(sqrt)((ok_float) mindim), factor;
	ok_float nrm_d = kZero, nrm_e = kZero;
	ok_float sqrt_n_over_m = MATH(sqrt)((ok_float) n / (ok_float) m);

	OK_RETURNIF_ERR( projector_normalization(W->P, &W->normalized) );
	OK_RETURNIF_ERR( projector_get_norm(W->P, &W->normA) );

	if (!(W->normalized)) {
		OK_CHECK_ERR( err, estimate_norm(linalg_handle, W, &W->normA) );
		W->normA /= sqrt_mindim;
 		OK_CHECK_ERR( err, W->operator_scale(W->A, kOne / W->normA) );
		W->normalized = 1;
	}

	OK_CHECK_ERR( err, blas_nrm2(linalg_handle, W->d, &nrm_d) );
	OK_CHECK_ERR( err, blas_nrm2(linalg_handle, W->e, &nrm_e) );
	factor = MATH(sqrt)(sqrt_n_over_m * nrm_d / nrm_e);

	if (factor == 0 || W->normA == 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIVIDE_BY_ZERO );

	OK_CHECK_ERR( err,
		vector_scale(W->d, kOne / (MATH(sqrt)(W->normA) * factor)) );
	OK_CHECK_ERR( err,
		vector_scale(W->e, factor / MATH(sqrt)(W->normA)) );

	return err;
}

POGS_PRIVATE ok_status update_problem(pogs_solver * solver, function_vector * f,
	function_vector * g)
{
	OK_CHECK_PTR(solver);
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_FNVECTOR(g);

	OK_RETURNIF_ERR( function_vector_memcpy_va(solver->f, f->objectives) );
	OK_RETURNIF_ERR( function_vector_memcpy_va(solver->g, g->objectives) );
	OK_RETURNIF_ERR( function_vector_div(solver->f, solver->W->d) );
	return OK_SCAN_ERR( function_vector_mul(solver->g, solver->W->e) );
}

POGS_PRIVATE ok_status initialize_variables(pogs_solver * solver)
{
	OK_CHECK_PTR(solver);

	pogs_variables * z = solver->z;
	operator * A = solver->W->A;
	if (solver->settings->x0 != OK_NULL) {
		vector_memcpy_va(z->temp->x, solver->settings->x0, 1);
		vector_div(z->temp->x, solver->W->e);
		A->apply(A->data, z->temp->x, z->temp->y);
		vector_memcpy_vv(z->primal->vec, z->temp->vec);
		vector_memcpy_vv(z->primal12->vec, z->temp->vec);
	}
	if (solver->settings->nu0 != OK_NULL) {
		vector_memcpy_va(z->dual->y, solver->settings->nu0, 1);
		vector_div(z->dual->y, solver->W->d);
		A->fused_adjoint(A->data, -kOne, z->dual->y, kZero, z->dual->x);
		vector_scale(z->dual->vec, -kOne / solver->rho);
	}
	return OPTKIT_SUCCESS;
}

/*
 * residual for primal feasibility
 *	||Ax^(k+1/2) - y^(k+1/2)||
 *
 * residual for dual feasibility
 * 	||A'yt^(k+1/2) + xt^(k+1/2)||
 */
POGS_PRIVATE ok_status update_residuals(void * linalg_handle,
	pogs_solver * solver, pogs_objectives * obj, pogs_residuals * res,
	pogs_tolerances * eps)
{
	if (!solver || !obj || !res || !eps)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	pogs_variables * z = solver->z;
	operator * A = solver->W->A;

	res->gap = obj->gap;

	vector_memcpy_vv(z->temp->y, z->primal12->y);
	A->fused_apply(A->data, kOne, z->primal12->x, -kOne, z->temp->y);
	blas_nrm2(linalg_handle, z->temp->y, &res->primal);

	vector_memcpy_vv(z->temp->x, z->dual12->x);
	A->fused_adjoint(A->data, kOne, z->dual12->y, kOne, z->temp->x);
	blas_nrm2(linalg_handle, z->temp->x, &res->dual);

	return OPTKIT_SUCCESS;
}

POGS_PRIVATE int check_convergence(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!solver || !obj || !res || !eps)
		err = OPTKIT_ERROR_UNALLOCATED;

	OK_CHECK_ERR( err, update_objective(linalg_handle, solver->f, solver->g,
		solver->rho, solver->z, obj) );
	OK_CHECK_ERR( err, update_tolerances(linalg_handle, solver->z, obj,
		eps) );
	OK_CHECK_ERR( err, update_residuals(linalg_handle, solver, obj, res,
		eps) );

	return !err && (res->primal < eps->primal) && (res->dual < eps->dual) &&
		(res->gap < eps->gap || !(solver->settings->gapstop));
}

/*
 * update z^{k+1} according to rule:
 *	( x^{k+1}, y^{k+1} ) =
 *		Proj_{y=Ax} (x^{k+1/2 + xt^k, y^{k+1/2} + yt^k)
 */
POGS_PRIVATE ok_status project_primal(void * linalg_handle, projector * proj,
	pogs_variables * z, ok_float alpha, ok_float tol)
{
	if (!proj || !proj->data || !z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	vector_set_all(z->temp->vec, kZero);
	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->temp->vec);
	return OK_SCAN_ERR( proj->project(proj->data, z->temp->x, z->temp->y,
		z->primal->x, z->primal->y, tol) );
}

POGS_PRIVATE ok_status pogs_solver_loop(pogs_solver * solver, pogs_info * info)
{
	if (!solver || !info)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	/* declare / get handles to all auxiliary types */
	int converged = 0;
	uint k, PRINT_ITER = 10000u;
	adapt_params rho_params = (adapt_params){kDELTAMIN, kZero, kZero, kOne};
	pogs_settings * settings = solver->settings;
	pogs_variables * z = solver->z;
	pogs_objectives obj = (pogs_objectives){OK_NAN, OK_NAN, OK_NAN};
	pogs_residuals res = (pogs_residuals){OK_NAN, OK_NAN, OK_NAN};
	pogs_tolerances eps = (pogs_tolerances){0, 0, 0, 0, 0, 0, 0, 0};
	ok_status err = initialize_conditions(&obj, &res, &eps, settings,
		solver->z->m, solver->z->n);

	void * linalg_handle = solver->linalg_handle;
	ok_float tol_proj = kProjectorTolInitial;

	/* TODO: SET TOLPROJ!!!!!! */

	if (settings->verbose == 0)
		PRINT_ITER = settings->maxiter * 2u;
	else
		for (k = 0; k < settings->verbose && PRINT_ITER > 1; ++k)
			PRINT_ITER /= 10;

	/* signal start of execution */
	if (settings->verbose > 0)
		print_header_string();

	/* iterate until converged, or error/maxiter reached */
	for (k = 1; !err && k <= settings->maxiter; ++k) {
		OK_CHECK_ERR( err,
			set_prev(z) );
		OK_CHECK_ERR( err,
			prox(linalg_handle, solver->f, solver->g, z,
				solver->rho) );
		OK_CHECK_ERR( err,
			project_primal(linalg_handle, solver->W->P, z,
				settings->alpha, tol_proj) );
		OK_CHECK_ERR( err,
			update_dual(linalg_handle, z, settings->alpha) );

		converged = check_convergence(linalg_handle, solver, &obj, &res,
			&eps);

		if ((k % PRINT_ITER == 0 || converged || k == settings->maxiter)
			&& settings->verbose)
			print_iter_string(&res, &eps, &obj, k);

		if (converged || k == settings->maxiter)
			break;

		if (!err && settings->adaptiverho)
			OK_CHECK_ERR( err,
				adaptrho(z, settings, &solver->rho, &rho_params,
					&res, &eps, k) );
	}

	if (!converged && k == settings->maxiter)
		printf("reached max iter = %u\n", k);

	/* update info */
	info->rho = solver->rho;
	info->obj = obj.primal;
	info->converged = converged;
	info->err = err;
	info->k = k;
	return err;
}

pogs_solver * pogs_init(operator * A, const int direct,
	const ok_float equil_norm)
{
	ok_status err = OPTKIT_SUCCESS;
	int normalize;
	pogs_solver * solver = OK_NULL;
	projector * P = OK_NULL;

	OK_TIMER t = tic();

	/* make solver variables */
	OK_CHECK_ERR( err,
		pogs_solver_alloc(&solver, A, direct) );

	/* equilibrate A as (D * A_equil * E) = A */
	OK_CHECK_ERR( err,
		equilibrate(solver->linalg_handle, solver->W, equil_norm) );

	/* make projector; normalize A; adjust d, e accordingly */
	if (!err) {
		P = solver->W->P;
		normalize = (int)(P->kind == OkProjectorDenseDirect);
		OK_CHECK_ERR( err,
			P->initialize(P->data, normalize) );
		OK_CHECK_ERR( err,
			normalize_DAE(solver->linalg_handle, solver->W) );
		solver->init_time = toc(t);
	}

	if (err)
		pogs_solver_free(solver);

	return solver;
}


ok_status pogs_solve(pogs_solver * solver, function_vector * f,
	function_vector * g, const pogs_settings * settings, pogs_info * info,
	pogs_output * output)
{
	if (!solver || !settings || !info || !output)
		return OPTKIT_ERROR_UNALLOCATED;
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_FNVECTOR(g);

	ok_status err = OPTKIT_SUCCESS;
	OK_TIMER t = tic();

	/* copy settings */
	OK_CHECK_ERR( err,
		update_settings(solver->settings, settings) );

	/* copy and scale function vectors */
	OK_CHECK_ERR( err,
		update_problem(solver, f, g) );

	/* get warm start variables */
	if (!err && settings->warmstart)
		OK_SCAN_ERR( initialize_variables(solver) );
	if ( !(settings->resume) )
		solver->rho = settings->rho;

	info->setup_time = toc(t);
	if (!(settings->warmstart || settings->resume))
		info->setup_time += solver->init_time;

	/* run solver */
	if (!err) {
		t = tic();
		OK_SCAN_ERR( pogs_solver_loop(solver, info) );
		info->solve_time = toc(t);
	}

	/* unscale output */
	OK_CHECK_ERR( err,
		copy_output(output, solver->z, solver->W->d, solver->W->e,
			solver->rho, settings->suppress) );
	return err;
}

ok_status pogs_finish(pogs_solver * solver, const int reset)
{
	ok_status err = OK_SCAN_ERR( pogs_solver_free(solver) );
	if (reset)
		OK_MAX_ERR( err, ok_device_reset() );
	return err;
}

ok_status pogs(operator * A, function_vector * f, function_vector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	const int direct, const ok_float equil_norm, const int reset)
{
	pogs_solver * solver = OK_NULL;
	solver = pogs_init(A, direct, equil_norm);
	OK_RETURNIF_ERR(
		pogs_solve(solver, f, g, settings, info, output) );
	return pogs_finish(solver, reset);
}


operator * pogs_dense_operator_gen(const ok_float * A, size_t m, size_t n,
	enum CBLAS_ORDER order)
{
	ok_status err = OPTKIT_SUCCESS;
	operator * o = OK_NULL;
	matrix * A_mat = OK_NULL;

	if (!A) {
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	} else  {
		ok_alloc(A_mat, sizeof(*A_mat));
		OK_CHECK_ERR( err, matrix_calloc(A_mat, m, n, order) );
		OK_CHECK_ERR( err, matrix_memcpy_ma(A_mat, A, order) );
		if (!err)
			o = dense_operator_alloc(A_mat);
	}
	return o;
}

operator * pogs_sparse_operator_gen(const ok_float * val, const ok_int * ind,
	const ok_int * ptr, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order)
{
	void * handle = OK_NULL;
	operator * o = OK_NULL;
	sp_matrix * A = OK_NULL;
	ok_status err = OPTKIT_SUCCESS;

	if (!val || !ind || !ptr)
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	else {
		ok_alloc(A, sizeof(*A));
		OK_CHECK_ERR( err,
			sp_matrix_calloc(A, m, n, nnz, order) );
		OK_CHECK_ERR( err,
			sp_make_handle(&handle) );
		OK_CHECK_ERR( err,
			sp_matrix_memcpy_ma(handle, A, val, ind, ptr) );
		OK_CHECK_ERR( err,
			sp_destroy_handle(handle) );
		if (!err)
			o = sparse_operator_alloc(A);
	}
	return o;
}

ok_status pogs_dense_operator_free(operator * A)
{
	OK_CHECK_OPERATOR(A);
	matrix * A_mat = dense_operator_get_matrix_pointer(A);
	ok_status err = A->free(A->data);
	OK_MAX_ERR( err, matrix_free(A_mat) );
	ok_free(A_mat);
	ok_free(A);
	return err;
}

ok_status pogs_sparse_operator_free(operator * A)
{
	OK_CHECK_OPERATOR(A);
	sp_matrix * A_mat = sparse_operator_get_matrix_pointer(A);
	ok_status err = A->free(A->data);
	OK_MAX_ERR( err, sp_matrix_free(A_mat) );
	ok_free(A_mat);
	ok_free(A);
	return err;
}

/*
pogs_solver * pogs_load_solver(ok_float * A_equil, ok_float * LLT_factorization,
	ok_float * d, ok_float * e, ok_float * z, ok_float * z12,
	ok_float * z_dual, ok_float * z_dual12, ok_float * z_prev, ok_float rho,
	size_t m, size_t n, enum CBLAS_ORDER ord)
{
	pogs_solver * solver = OK_NULL;
	pogs_solver_alloc(&solver, m , n, ord);

	matrix_memcpy_ma(solver->W->A, A_equil, ord);

	#ifndef OPTKIT_INDIRECT
	matrix_memcpy_ma(solver->W->P->L, LLT_factorization, ord);
	#endif

	vector_memcpy_va(solver->W->d, d, 1);
	vector_memcpy_va(solver->W->e, e, 1);

	vector_memcpy_va(solver->z->primal->vec, z, 1);
	vector_memcpy_va(solver->z->primal12->vec, z12, 1);
	vector_memcpy_va(solver->z->dual->vec, z_dual, 1);
	vector_memcpy_va(solver->z->dual12->vec, z_dual12, 1);
	vector_memcpy_va(solver->z->prev->vec, z_prev, 1);

	solver->rho = rho;

	return solver;
}

void pogs_extract_solver(pogs_solver * solver, ok_float * A_equil,
	ok_float * LLT_factorization, ok_float * d, ok_float * e, ok_float * z,
	ok_float * z12, ok_float * z_dual, ok_float * z_dual12,
	ok_float * z_prev, ok_float * rho, enum CBLAS_ORDER ord)
{
	matrix_memcpy_am(A_equil, solver->W->A, ord);

	#ifndef OPTKIT_INDIRECT
	matrix_memcpy_am(LLT_factorization, solver->W->P->L, ord);
	#endif

	vector_memcpy_av(d, solver->W->d, 1);
	vector_memcpy_av(e, solver->W->e, 1);

	vector_memcpy_av(z, solver->z->primal->vec, 1);
	vector_memcpy_av(z12, solver->z->primal12->vec, 1);
	vector_memcpy_av(z_dual, solver->z->dual->vec, 1);
	vector_memcpy_av(z_dual12, solver->z->dual12->vec, 1);
	vector_memcpy_av(z_prev, solver->z->prev->vec, 1);

	*rho = solver->rho;
}
*/

#ifdef __cplusplus
}
#endif
