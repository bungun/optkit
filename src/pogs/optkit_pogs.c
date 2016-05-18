#include "optkit_pogs.h"

#ifdef __cplusplus
extern "C" {
#endif

int is_direct()
{
	return sizeof(projector_) == sizeof(direct_projector);
}

POGS_PRIVATE ok_status pogs_matrix_alloc(pogs_matrix ** M, size_t m, size_t n,
	enum CBLAS_ORDER ord)
{
	if (*M != OK_NULL)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_status err = OPTKIT_SUCCESS;
	pogs_matrix * M_ = OK_NULL;
	ok_alloc(M_, sizeof(*M_));
	ok_alloc(M_->A, sizeof(*M_->A));
	OK_CHECK_ERR( err, matrix_calloc(M_->A, m, n, ord) );
	ok_alloc(M_->P, sizeof(projector_));
	OK_CHECK_ERR( err, PROJECTOR(alloc)(M_->P, M_->A) );
	ok_alloc(M_->d, sizeof(*M_->d));
	ok_alloc(M_->e, sizeof(*M_->e));
	OK_CHECK_ERR( err, vector_calloc(M_->d, m) );
	OK_CHECK_ERR( err, vector_calloc(M_->e, n) );
	M_->skinny = (m >= n);
	M_->normalized = 0;
	M_->equilibrated = 0;
	if (err)
		OK_MAX_ERR( err, pogs_matrix_free(M_) );
	else
		*M = M_;
	return err;
}

POGS_PRIVATE ok_status pogs_matrix_free(pogs_matrix * M)
{
	if (!M || !M->A)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	ok_status err = OK_SCAN_ERR( PROJECTOR(free)(M->P) );
	ok_free(M->P);
	OK_MAX_ERR( err, matrix_free(M->A) );
	OK_MAX_ERR( err, vector_free(M->d) );
	OK_MAX_ERR( err, vector_free(M->e) );
	ok_free(M);
	return err;
}

POGS_PRIVATE ok_status pogs_solver_alloc(pogs_solver ** solver, size_t m,
	size_t n, enum CBLAS_ORDER ord)
{
	ok_status err = OPTKIT_SUCCESS;
	pogs_solver * s = OK_NULL;

	if (*solver)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_alloc(s, sizeof(*s));
	ok_alloc(s->settings, sizeof(*s->settings));
	err = set_default_settings(s->settings);
	ok_alloc(s->f, sizeof(*s->f));
	ok_alloc(s->g, sizeof(*s->g));
	OK_CHECK_ERR( err, function_vector_calloc(s->f, m) );
	OK_CHECK_ERR( err, function_vector_calloc(s->g, n) );
	OK_CHECK_ERR( err, pogs_variables_alloc(&(s->z), m, n) );
	OK_CHECK_ERR( err, pogs_matrix_alloc(&(s->M), m, n, ord) );
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
	OK_MAX_ERR( err, pogs_matrix_free(solver->M) );
	OK_MAX_ERR( err, pogs_variables_free(solver->z) );
	ok_free(solver->settings);
	OK_MAX_ERR( err, function_vector_free(solver->f) );
	OK_MAX_ERR( err, function_vector_free(solver->g) );
	ok_free(solver->f);
	ok_free(solver->g);
	ok_free(solver);
	return err;
}

POGS_PRIVATE ok_status equilibrate(void * linalg_handle, ok_float * A_orig,
	pogs_matrix * M, enum CBLAS_ORDER ord)
{
	ok_status err = regularized_sinkhorn_knopp(linalg_handle, A_orig, M->A,
		M->d, M->e, ord);
	M->equilibrated = (err == OPTKIT_SUCCESS);
	return err;
}

/* STUB */
POGS_PRIVATE ok_status estimate_norm(void * linalg_handle, pogs_matrix * M,
	ok_float * normest)
{
	*normest = kOne;
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status normalize_DAE(void * linalg_handle, pogs_matrix * M)
{
	OK_CHECK_PTR(M);

	ok_status err = OPTKIT_SUCCESS;
	size_t m = M->A->size1,  n = M->A->size2;
	size_t mindim = m < n ? m : n;
	ok_float factor;
	ok_float nrm_d = kZero, nrm_e = kZero;
	ok_float sqrt_n_over_m = MATH(sqrt)((ok_float) n / (ok_float) m);

	M->normalized = M->P->normalized;
	M->normA = M->P->normA;

	if (!(M->normalized)) {
		OK_CHECK_ERR( err, estimate_norm(linalg_handle, M, &M->normA) );
		M->normA /=  MATH(sqrt)((ok_float) mindim);
		OK_CHECK_ERR( err, matrix_scale(M->A, kOne / M->normA) );
		M->normalized = 1;
	}
	OK_CHECK_ERR( err, blas_nrm2(linalg_handle, M->d, &nrm_d) );
	OK_CHECK_ERR( err, blas_nrm2(linalg_handle, M->e, &nrm_e) );
	factor = MATH(sqrt)(sqrt_n_over_m * nrm_d / nrm_e);

	if (factor == 0 || M->normA == 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIVIDE_BY_ZERO );

	OK_CHECK_ERR( err,
		vector_scale(M->d, kOne / (MATH(sqrt)(M->normA) * factor)) );
	OK_CHECK_ERR( err,
		vector_scale(M->e, factor / MATH(sqrt)(M->normA)) );
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
	OK_RETURNIF_ERR( function_vector_div(solver->f, solver->M->d) );
	return OK_SCAN_ERR( function_vector_mul(solver->g, solver->M->e) );
}

POGS_PRIVATE ok_status initialize_variables(pogs_solver * solver)
{
	OK_CHECK_PTR(solver);
	pogs_variables * z = solver->z;

	if (solver->settings->x0 != OK_NULL) {
		vector_memcpy_va(z->temp->x, solver->settings->x0, 1);
		vector_div(z->temp->x, solver->M->e);
		blas_gemv(solver->linalg_handle, CblasNoTrans, kOne,
			solver->M->A, z->temp->x, kZero, z->temp->y);
		vector_memcpy_vv(z->primal->vec, z->temp->vec);
		vector_memcpy_vv(z->primal12->vec, z->temp->vec);
	}
	if (solver->settings->nu0 != OK_NULL) {
		vector_memcpy_va(z->dual->y, solver->settings->nu0, 1);
		vector_div(z->dual->y, solver->M->d);
		blas_gemv(solver->linalg_handle, CblasTrans, -kOne,
			solver->M->A, z->dual->y, kZero, z->dual->x);
		vector_scale(z->dual->vec, -kOne / solver->rho);
	}
	return OPTKIT_SUCCESS;
}

/*
 * residual for primal feasibility
 *	||Ax^(k+1/2) - y^(k+1/2)||
 *
 * residual for dual feasibility
 * 	||A'yt^(k+1/2) - xt^(k+1/2)||
 */
POGS_PRIVATE ok_status update_residuals(void * linalg_handle,
	pogs_solver * solver, pogs_objectives * obj, pogs_residuals * res,
	pogs_tolerances * eps)
{
	if (!solver || !obj || !res || !eps)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	pogs_variables * z = solver->z;
	matrix * A = solver->M->A;

	res->gap = obj->gap;

	vector_memcpy_vv(z->temp->y, z->primal12->y);
	blas_gemv(linalg_handle, CblasNoTrans, kOne, A, z->primal12->x, -kOne,
		z->temp->y);
	blas_nrm2(linalg_handle, z->temp->y, &res->primal);

	vector_memcpy_vv(z->temp->x, z->dual12->x);
	blas_gemv(linalg_handle, CblasTrans, kOne, A, z->dual12->y, kOne,
		z->temp->x);
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
POGS_PRIVATE ok_status project_primal(void * linalg_handle, projector_ * proj,
	pogs_variables * z,  ok_float alpha)
{
	if (!proj || !z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	vector_set_all(z->temp->vec, kZero);
	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->temp->vec);
	return OK_SCAN_ERR( PROJECTOR(project)(linalg_handle, proj, z->temp->x,
		z->temp->y, z->primal->x, z->primal->y) );
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
			project_primal(linalg_handle, solver->M->P, z,
				settings->alpha) );
		OK_CHECK_ERR( err,
			update_dual(linalg_handle, z, settings->alpha) );

		converged = check_convergence(linalg_handle, solver, &obj, &res,
			&eps);

		if ((k % PRINT_ITER == 0 || converged ||k == settings->maxiter)
			&& settings->verbose)
			print_iter_string(&res, &eps, &obj, k);

		if (converged || k == settings->maxiter)
			break;

		if (settings->adaptiverho)
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

pogs_solver * pogs_init(ok_float * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	ok_status err = OPTKIT_SUCCESS;
	pogs_solver * solver = OK_NULL;
	pogs_matrix * M = OK_NULL;
	OK_TIMER t = tic();

	/* make variables, matrix */
	err = pogs_solver_alloc(&solver, m , n, ord);

	/* equilibrate A as (D * A_equil * E) = A */
	OK_CHECK_ERR( err,
		equilibrate(solver->linalg_handle, A, solver->M, ord) );

	/* make projector; normalize A; adjust d, e accordingly */
	if (!err) {
		M = solver->M;
		OK_CHECK_ERR( err,
			PROJECTOR(initialize)(solver->linalg_handle, M->P, 1) );
		OK_CHECK_ERR( err,
			normalize_DAE(solver->linalg_handle, M) );
	}

	solver->init_time = toc(t);
	if (err)
		pogs_solver_free(solver);

	return solver;
}

ok_status pogs_solve(pogs_solver * solver, function_vector * f,
	function_vector * g, const pogs_settings * settings, pogs_info * info,
	pogs_output * output)
{
	if (!solver || !settings || !info || !output)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
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
		OK_CHECK_ERR( err, pogs_solver_loop(solver, info) );
		info->solve_time = toc(t);
	}

	/* unscale output */
	OK_CHECK_ERR( err,
		copy_output(output, solver->z, solver->M->d, solver->M->e,
			solver->rho, settings->suppress) );
	return err;
}

ok_status pogs_finish(pogs_solver * solver, int reset)
{
	ok_status err = OK_SCAN_ERR( pogs_solver_free(solver) );
	if (reset)
		OK_MAX_ERR( err, ok_device_reset() );
	return err;
}

ok_status pogs(ok_float * A, function_vector * f, function_vector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	enum CBLAS_ORDER ord, int reset)
{
	pogs_solver * solver = OK_NULL;
	solver = pogs_init(A, f->size, g->size, ord);
	OK_RETURNIF_ERR( pogs_solve(solver, f, g, settings, info, output) );
	return pogs_finish(solver, reset);
}

pogs_solver * pogs_load_solver(ok_float * A_equil, ok_float * LLT_factorization,
	ok_float * d, ok_float * e, ok_float * z, ok_float * z12,
	ok_float * z_dual, ok_float * z_dual12, ok_float * z_prev, ok_float rho,
	size_t m, size_t n, enum CBLAS_ORDER ord)
{
	ok_status err;
	pogs_solver * solver = OK_NULL;
	err = pogs_solver_alloc(&solver, m , n, ord);

	OK_CHECK_ERR( err,
		matrix_memcpy_ma(solver->M->A, A_equil, ord) );

	#ifndef OPTKIT_INDIRECT
	OK_CHECK_ERR( err,
		matrix_memcpy_ma(solver->M->P->L, LLT_factorization, ord) );
	#endif

	OK_CHECK_ERR( err,
		vector_memcpy_va(solver->M->d, d, 1) );
	OK_CHECK_ERR( err,
		vector_memcpy_va(solver->M->e, e, 1) );
	OK_CHECK_ERR( err,
		vector_memcpy_va(solver->z->primal->vec, z, 1) );
	OK_CHECK_ERR( err,
		vector_memcpy_va(solver->z->primal12->vec, z12, 1) );
	OK_CHECK_ERR( err,
		vector_memcpy_va(solver->z->dual->vec, z_dual, 1) );
	OK_CHECK_ERR( err,
		vector_memcpy_va(solver->z->dual12->vec, z_dual12, 1) );
	OK_CHECK_ERR( err,
		vector_memcpy_va(solver->z->prev->vec, z_prev, 1) );

	solver->rho = rho;
	if (err)
		pogs_solver_free(solver);

	return solver;
}

ok_status pogs_extract_solver(pogs_solver * solver, ok_float * A_equil,
	ok_float * LLT_factorization, ok_float * d, ok_float * e, ok_float * z,
	ok_float * z12, ok_float * z_dual, ok_float * z_dual12,
	ok_float * z_prev, ok_float * rho, enum CBLAS_ORDER ord)
{
	if (!solver)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	OK_RETURNIF_ERR( matrix_memcpy_am(A_equil, solver->M->A, ord) );

	#ifndef OPTKIT_INDIRECT
	OK_RETURNIF_ERR( matrix_memcpy_am(LLT_factorization, solver->M->P->L,
		ord) );
	#endif

	OK_RETURNIF_ERR( vector_memcpy_av(d, solver->M->d, 1) );
	OK_RETURNIF_ERR( vector_memcpy_av(e, solver->M->e, 1) );

	OK_RETURNIF_ERR( vector_memcpy_av(z, solver->z->primal->vec, 1) );
	OK_RETURNIF_ERR( vector_memcpy_av(z12, solver->z->primal12->vec, 1) );
	OK_RETURNIF_ERR( vector_memcpy_av(z_dual, solver->z->dual->vec, 1) );
	OK_RETURNIF_ERR( vector_memcpy_av(z_dual12, solver->z->dual12->vec,
		1) );
	OK_RETURNIF_ERR( vector_memcpy_av(z_prev, solver->z->prev->vec, 1) );

	*rho = solver->rho;
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
