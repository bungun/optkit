#include "optkit_pogs.h"

#ifdef __cplusplus
extern "C" {
#endif

ok_pogs_impl get_pogs_impl(void)
{
	return OK_POGS_TYPE;
}

ok_status pogs_work_alloc(pogs_work *W, pogs_solver_data *A,
	const pogs_solver_flags *flags)
{
	if (!A || !flags)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR( err, POGS(problem_data_alloc)(W, A, flags) );
	ok_alloc(W->d, sizeof(*W->d));
	ok_alloc(W->e, sizeof(*W->e));
	OK_CHECK_ERR( err, vector_calloc(W->d, W->A->size1) );
	OK_CHECK_ERR( err, vector_calloc(W->e, W->A->size2) );
	W->skinny = (W->A->size1 >= W->A->size2);
	W->normalized = 0;
	W->equilibrated = 0;
	OK_CHECK_ERR( err, blas_make_handle(&(W->linalg_handle)) );
	if (err)
		OK_MAX_ERR( err, pogs_work_free(W) );
	return err;
}

ok_status pogs_work_free(pogs_work *W)
{
	OK_CHECK_PTR(W);
	ok_status err = OK_SCAN_ERR( POGS(problem_data_free)(W) );
	OK_MAX_ERR( err, vector_free(W->d) );
	OK_MAX_ERR( err, vector_free(W->e) );
	ok_free(W->d);
	ok_free(W->e);
	OK_MAX_ERR( err, blas_destroy_handle(W->linalg_handle) );
	return err;
}

ok_status pogs_solver_alloc(pogs_solver *solver, pogs_solver_data *A,
	const pogs_solver_flags *flags)
{
	size_t m, n;
        ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(solver);
	if (solver->W)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	ok_alloc(solver->settings, sizeof(*solver->settings));
	OK_CHECK_ERR( err, pogs_set_default_settings(solver->settings) );
	ok_alloc(solver->W, sizeof(*solver->W));
	OK_CHECK_ERR( err, pogs_work_alloc(solver->W, A, flags) );
	if (!err) {
		m = solver->W->A->size1;
		n = solver->W->A->size2;
	}
	ok_alloc(solver->f, sizeof(*solver->f));
	ok_alloc(solver->g, sizeof(*solver->g));
	OK_CHECK_ERR( err, function_vector_calloc(solver->f, m) );
	OK_CHECK_ERR( err, function_vector_calloc(solver->g, n) );
	ok_alloc(solver->z, sizeof(*solver->z));
	OK_CHECK_ERR( err, pogs_variables_alloc(solver->z, m, n) );
	OK_CHECK_ERR( err, blas_make_handle(&(solver->linalg_handle)) );
	ok_alloc(solver->aa, sizeof(*solver->aa));
	// OK_CHECK_ERR( err, anderson_accelerator_init(solver->aa,
	// 	solver->z->fixed_point_iterate->size,
	// 	(size_t) solver->settings->anderson_lookback) );
	if (err)
		OK_MAX_ERR( err, pogs_solver_free(solver) );
	return err;
}

ok_status pogs_solver_free(pogs_solver *solver)
{
	OK_CHECK_PTR(solver);
	ok_status err = OK_SCAN_ERR( blas_destroy_handle(solver->linalg_handle) );
	OK_MAX_ERR( err, pogs_work_free(solver->W) );
	ok_free(solver->W);
	OK_MAX_ERR( err, pogs_variables_free(solver->z) );
	ok_free(solver->z);
	ok_free(solver->settings);
	OK_MAX_ERR( err, function_vector_free(solver->f) );
	OK_MAX_ERR( err, function_vector_free(solver->g) );
	ok_free(solver->f);
	ok_free(solver->g);
	// OK_MAX_ERR( err, anderson_accelerator_free(solver->aa) );
	ok_free(solver->aa);
	return err;
}

/*
 * For scaled problem
 *
 *	min. f(D^{-1}y') + g(Ex')
 *	s.t. y' = DAEx'
 *
 * (with y' = Dy, x' = E^{-1}x), normalize D, A, and E such that:
 *
 * 	A' = DAE / ||DAE||_2
 * 	D' = (D / sqrt(||DAE||_2) / (sqrt(n/m) * ||D||_2/||E||_2)
 * 	E' = (E / sqrt(||DAE||_2) * (sqrt(n/m) * ||D||_2/||E||_2)
 *
 * to obtain normalized problem
 *
 *	min. f(D'^{-1}y") + g(E'x")
 *	s.t. y" = A'x"
 *
 * (with y" = D'y, x" = E^{-1}x). To verify y = Ax
 *
 &	y" = A'x"
 *	y = D'^{-1}A'E'^{-1}x
 *	y = D'^{-1}(DAE/||DAE||_2)E'^{-1}x
 *	y = D^{-1} * sqrt(||DAE||_2) * (sqrt(n/m) * ||D||_2/||E||_2) *
 *		DAE / ||DAE||_2 *
 *		E^{-1} * sqrt(||DAE||_2) / (sqrt(n/m) * ||D||_2/||E||_2) x
 *	y = Ax
 */
ok_status pogs_normalize_DAE(pogs_work *W)
{
	OK_CHECK_PTR(W);

	ok_status err = OPTKIT_SUCCESS;
	size_t m = W->A->size1,  n = W->A->size2;
	size_t mindim = m < n ? m : n;
	ok_float sqrt_mindim = MATH(sqrt)((ok_float) mindim);
	ok_float factor;
	ok_float nrm_d = kZero, nrm_e = kZero;
	ok_float sqrt_n_over_m = MATH(sqrt)((ok_float) n / (ok_float) m);

	OK_CHECK_ERR( err, POGS(work_get_norm)(W) );
	if (!(W->normalized)) {
		OK_CHECK_ERR( err, POGS(estimate_norm)(W, &W->normA) );
		W->normA /= sqrt_mindim;
 		OK_CHECK_ERR( err, POGS(work_normalize)(W) );
		W->normalized = 1;
	}

	OK_CHECK_ERR( err, blas_nrm2(W->linalg_handle, W->d, &nrm_d) );
	OK_CHECK_ERR( err, blas_nrm2(W->linalg_handle, W->e, &nrm_e) );
	factor = MATH(sqrt)(sqrt_n_over_m *nrm_d / nrm_e);

	if (factor == 0 || W->normA == 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIVIDE_BY_ZERO );

	OK_CHECK_ERR( err,
		vector_scale(W->d, kOne / (MATH(sqrt)(W->normA) * factor)) );
	OK_CHECK_ERR( err,
		vector_scale(W->e, factor / MATH(sqrt)(W->normA)) );
	return err;
}

ok_status pogs_set_z0(pogs_solver *solver)
{
	OK_CHECK_PTR(solver);

	ok_status err = OPTKIT_SUCCESS;
	pogs_variables *z = solver->z;

	if (solver->settings->x0 != OK_NULL) {
		OK_CHECK_ERR( err, vector_memcpy_va(z->temp->x,
			solver->settings->x0, 1) );
		OK_CHECK_ERR( err, vector_div(z->temp->x, solver->W->e) );
		OK_CHECK_ERR( err, POGS(apply_matrix)(solver->W, kOne,
			z->temp->x, kZero, z->temp->y) );
		OK_CHECK_ERR( err, pogs_graph_vector_copy(z->primal, z->temp) );
		OK_CHECK_ERR( err, pogs_graph_vector_copy(z->primal12, z->temp) );

	}
	if (solver->settings->nu0 != OK_NULL) {
		OK_CHECK_ERR( err, vector_memcpy_va(z->dual->y,
			solver->settings->nu0, 1) );
		OK_CHECK_ERR( err, vector_div(z->dual->y, solver->W->d) );
		OK_CHECK_ERR( err, POGS(apply_adjoint)(
			solver->W, -kOne, z->dual->y, kZero, z->dual->x) );
		OK_CHECK_ERR( err, vector_scale(z->dual->vec,
			-kOne / solver->rho) );
	}
	return err;
}

/* z^k <- z^{k+1} */
ok_status pogs_primal_update(pogs_variables *z)
{
	return OK_SCAN_ERR( pogs_graph_vector_copy(z->prev, z->primal) );
}

/*
 * update z^{k + 1/2} according to following rule:
 *
 *	y^{k+1/2} = Prox_{rho, f} (y^k - yt^k)
 *	x^{k+1/2} = Prox_{rho, g} (x^k - xt^k)
 */
ok_status pogs_prox(void *linalg_handle, function_vector *f,
	function_vector *g, pogs_variables *z, ok_float rho)
{
	OK_RETURNIF_ERR( pogs_graph_vector_copy(z->temp, z->primal) );
	OK_RETURNIF_ERR(
		blas_axpy(linalg_handle, -kOne, z->dual->vec, z->temp->vec) );
	OK_RETURNIF_ERR(
		prox_eval_vector(f, rho, z->temp->y, z->primal12->y) );
	return OK_SCAN_ERR(
		prox_eval_vector(g, rho, z->temp->x, z->primal12->x));
}

/*
 * update z^{k+1} according to rule:
 *	( x^{k+1}, y^{k+1} ) =
 *		Proj_{y=Ax} (x^{k+1/2 + xt^k, y^{k+1/2} + yt^k)
 *
 * overrelaxation variant:
 *	instead of 	z^{k+1} = proj(z^{k+1/2} + zt^k),
 *	perform		z^{k+1} = proj(alpha * z^{k+1/2} + (1-alpha)z^k + zt^k)
 */
ok_status pogs_project_graph(pogs_work *W, pogs_variables *z, ok_float alpha,
	ok_float tol)
{
	ok_status err = OPTKIT_SUCCESS;
	void *linalg_handle = W->linalg_handle;
	if (!W || !z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_ERR( err, vector_set_all(z->temp->vec, kZero) );
	OK_CHECK_ERR( err, blas_axpy(
		linalg_handle, alpha, z->primal12->vec, z->temp->vec) );
	OK_CHECK_ERR( err, blas_axpy(
		linalg_handle, kOne - alpha, z->prev->vec, z->temp->vec) );
	OK_CHECK_ERR( err, blas_axpy(
		linalg_handle, kOne, z->dual->vec, z->temp->vec) );
	OK_CHECK_ERR( err, POGS(project_graph)(W, z->temp->x,
		z->temp->y, z->primal->x, z->primal->y, tol) );
	return err;
}

/*
 * update zt^{k+1/2} and zt^{k+1} according to:
 *
 * 	zt^{k+1/2} = zt^k + z^{k+1/2} - z^k
 * 	zt^{k+1}   = zt^k + z^{k+1/2} - z^k+1
 *
 * overrelaxation variant:
 *	instead of above update, perform
 *		zt^{k+1} = zt^k + alpha * z^{k+1/2} + (1-alpha)z^k - z^k+1
 */
ok_status pogs_dual_update(void *linalg_handle, pogs_variables *z,
	ok_float alpha)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(z);

	OK_CHECK_ERR( err, pogs_graph_vector_copy(z->dual12, z->primal12) );
	OK_CHECK_ERR( err, blas_axpy(linalg_handle, -kOne, z->prev->vec, z->dual12->vec) );
	OK_CHECK_ERR( err, blas_axpy(linalg_handle, kOne, z->dual->vec, z->dual12->vec) );

	OK_CHECK_ERR( err, blas_axpy(linalg_handle, alpha, z->primal12->vec, z->dual->vec) );
	OK_CHECK_ERR( err, blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->dual->vec) );
	OK_CHECK_ERR( err, blas_axpy(linalg_handle, -kOne, z->primal->vec, z->dual->vec) );

	return OPTKIT_SUCCESS;
}

ok_status pogs_accelerate(pogs_solver *solver)
{
	OK_CHECK_PTR(solver);
	if (solver->settings->accelerate)
		OK_RETURNIF_ERR( ACCELERATOR(accelerate)(
			solver->aa, solver->z->fixed_point_iterate) );
	return OPTKIT_SUCCESS;
}

ok_status pogs_iterate(pogs_solver *solver)
{
	ok_status err = OPTKIT_SUCCESS;
	pogs_solver *s = solver;
	ok_float alpha = s->settings->alpha;
	ok_float tolproj = s->settings->tolproj;
	err = OK_SCAN_ERR( pogs_primal_update(s->z) );
	OK_CHECK_ERR( err, pogs_prox(s->linalg_handle, s->f, s->g, s->z, s->rho) );
	OK_CHECK_ERR( err, pogs_project_graph(s->W, s->z, alpha, tolproj) );
	OK_CHECK_ERR( err, pogs_dual_update(s->linalg_handle, s->z, alpha) );
	return err;
	/* TODO: modify tolproj as monotonic decr function of iteration? */
}

/*
 * residual for primal feasibility
 *	||Ax^(k+1/2) - y^(k+1/2)||
 *
 * residual for dual feasibility
 * 	||A'yt^(k+1/2) - xt^(k+1/2)||
 */
ok_status pogs_update_residuals(pogs_solver *solver,
	pogs_objective_values *obj, pogs_residuals *res)
{
	if (!solver || !obj || !res)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	ok_status err = OPTKIT_SUCCESS;
	pogs_variables *z = solver->z;
	void *linalg_handle = solver->linalg_handle;

	res->gap = obj->gap;
	OK_CHECK_ERR( err, vector_memcpy_vv(z->temp->y, z->primal12->y) );
	OK_CHECK_ERR( err, POGS(apply_matrix)(solver->W, kOne,
		z->primal12->x, -kOne, z->temp->y) );
	OK_CHECK_ERR( err, blas_nrm2(linalg_handle, z->temp->y, &res->primal) );
	OK_CHECK_ERR( err, vector_memcpy_vv(z->temp->x, z->dual12->x) );
	OK_CHECK_ERR( err, POGS(apply_adjoint)(solver->W, kOne,
		z->dual12->y, kOne, z->temp->x) );
	OK_CHECK_ERR( err, blas_nrm2(linalg_handle, z->temp->x, &res->dual) );
	return err;
}

ok_status pogs_check_convergence(pogs_solver *solver,
	pogs_objective_values *obj, pogs_residuals *res,
	pogs_tolerances *tol, int *converged)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!solver || !obj || !res || !tol || !converged)
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	pogs_solver *s = solver;
	OK_CHECK_ERR( err, pogs_update_objective_values(
		s->linalg_handle, s->f, s->g, s->rho, s->z, obj) );
	OK_CHECK_ERR( err, pogs_update_tolerances(
		s->linalg_handle, s->z, obj, tol) );
	OK_CHECK_ERR( err, pogs_update_residuals(s, obj, res) );

	*converged = !err &&
		(res->primal < tol->primal) &&
		(res->dual < tol->dual) &&
		((res->gap < tol->gap) || !(solver->settings->gapstop));
	return err;
}

ok_status pogs_setup_diagnostics(pogs_solver *solver, const uint iters)
{
	OK_CHECK_PTR(solver);
	if (solver->convergence)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_status err = OPTKIT_SUCCESS;
	ok_alloc(solver->convergence, sizeof(*solver->convergence));
	ok_alloc(solver->convergence->primal, sizeof(*solver->convergence->primal));
	ok_alloc(solver->convergence->dual, sizeof(*solver->convergence->dual));
	ok_alloc(solver->convergence->primal_tol, sizeof(*solver->convergence->primal_tol));
	ok_alloc(solver->convergence->dual_tol, sizeof(*solver->convergence->dual_tol));
	OK_CHECK_ERR( err, vector_calloc(solver->convergence->primal, (size_t) iters) );
	OK_CHECK_ERR( err, vector_calloc(solver->convergence->dual, (size_t) iters) );
	OK_CHECK_ERR( err, vector_calloc(solver->convergence->primal_tol, (size_t) iters) );
	OK_CHECK_ERR( err, vector_calloc(solver->convergence->dual_tol, (size_t) iters) );
	return err;
}

ok_status pogs_record_diagnostics(pogs_solver *solver,
	const pogs_residuals *res, const pogs_tolerances *tol, const uint iter)
{
	ok_status err = OPTKIT_SUCCESS;
	vector v = (vector){OK_NULL};
	OK_CHECK_ERR( err, vector_subvector(&v, solver->convergence->primal,
		iter - 1, 1) );
	OK_CHECK_ERR( err, vector_set_all(&v, res->primal) );
	OK_CHECK_ERR( err, vector_subvector(&v, solver->convergence->dual,
		iter - 1, 1) );
	OK_CHECK_ERR( err, vector_set_all(&v, res->dual) );
	OK_CHECK_ERR( err, vector_subvector(&v, solver->convergence->primal_tol,
		iter - 1, 1) );
	OK_CHECK_ERR( err, vector_set_all(&v, tol->primal) );
	OK_CHECK_ERR( err, vector_subvector(&v, solver->convergence->dual_tol,
		iter - 1, 1) );
	OK_CHECK_ERR( err, vector_set_all(&v, tol->dual) );
	return err;
}

ok_status pogs_emit_diagnostics(pogs_output *output, pogs_solver *solver)
{
	if (!output || !solver || !solver->convergence)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR( err, vector_memcpy_av(output->primal_residuals,
		solver->convergence->primal, 1) );
	OK_CHECK_ERR( err, vector_memcpy_av(output->dual_residuals,
		solver->convergence->dual, 1) );
	OK_CHECK_ERR( err, vector_memcpy_av(output->primal_tolerances,
		solver->convergence->primal_tol, 1) );
	OK_CHECK_ERR( err, vector_memcpy_av(output->dual_tolerances,
		solver->convergence->dual_tol, 1) );
	OK_MAX_ERR( err, vector_free(solver->convergence->primal) );
	OK_MAX_ERR( err, vector_free(solver->convergence->dual) );
	OK_MAX_ERR( err, vector_free(solver->convergence->primal_tol) );
	OK_MAX_ERR( err, vector_free(solver->convergence->dual_tol) );
	ok_free(solver->convergence->primal);
	ok_free(solver->convergence->dual);
	ok_free(solver->convergence->primal_tol);
	ok_free(solver->convergence->dual_tol);
	ok_free(solver->convergence);
	return err;
}


ok_status pogs_solver_loop(pogs_solver *solver, pogs_info *info)
{
	if (!solver || !info)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	/* declare / get handles to all auxiliary types */
	ok_status err = OPTKIT_SUCCESS;
	int converged = 0;
	uint k, PRINT_ITER = 10000u;
	adapt_params rho_params = (adapt_params){kZero, kZero, kZero, kZero};
	pogs_settings *settings = solver->settings;
	pogs_objective_values obj = (pogs_objective_values){
		OK_NAN, OK_NAN, OK_NAN};
	pogs_residuals res = (pogs_residuals){OK_NAN, OK_NAN, OK_NAN};
	pogs_tolerances tol = (pogs_tolerances){0, 0, 0, 0, 0, 0, 0, 0};

	err = OK_SCAN_ERR( pogs_initialize_conditions(&obj, &res,
		&tol, settings, solver->z->m, solver->z->n) );
	OK_CHECK_ERR( err, pogs_adaptive_rho_initialize(&rho_params) );

	// TODO: SET ANDERSON LOOKBACK WINDOW

	OK_CHECK_ERR( err, pogs_set_print_iter(&PRINT_ITER, settings) );

	/* signal start of execution */
	if (settings->verbose > 0)
		OK_CHECK_ERR(err, pogs_print_header_string() );

	/* iterate until converged, or error/maxiter reached */
	for (k = 1; !err && k <= settings->maxiter; ++k) {
		OK_CHECK_ERR( err, pogs_iterate(solver) );
		OK_CHECK_ERR( err, pogs_accelerate(solver) );
		OK_CHECK_ERR( err, pogs_check_convergence(solver, &obj, &res,
			&tol, &converged) );

		if ((k % PRINT_ITER == 0 || converged || k == settings->maxiter)
			&& settings->verbose)
			pogs_print_iter_string(&res, &tol, &obj, k);

		if (settings->diagnostic)
			OK_CHECK_ERR( err, pogs_record_diagnostics(solver,
				&res, &tol, k) );

		if (converged || k == settings->maxiter)
			break;

		OK_CHECK_ERR( err, pogs_adapt_rho(solver->z, &(solver->rho),
			&rho_params, settings, &res, &tol, k) );
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

pogs_solver * pogs_init(pogs_solver_data *A, const pogs_solver_flags *flags)
{
	ok_status err = OPTKIT_SUCCESS;
	pogs_solver *solver = OK_NULL;

	OK_TIMER t = ok_timer_tic();
	ok_alloc(solver, sizeof(*solver));
	OK_CHECK_ERR( err, pogs_solver_alloc(solver, A, flags) );
	OK_CHECK_ERR( err, POGS(equilibrate_matrix)(solver->W, A, flags) );
	OK_CHECK_ERR( err, POGS(initalize_graph_projector)(
		solver->W) );
	OK_CHECK_ERR( err, pogs_normalize_DAE(solver->W) );
	solver->init_time = ok_timer_toc(t);

	if (err) {
		OK_MAX_ERR( err, pogs_solver_free(solver) );
		ok_free(solver);
	}
	return solver;
}

/* TODO: MAKE STATE VECTOR A SINGLE ALLOCATION (DONE) */
/* TODO: MAKE FIXED POINT ITERATE A VIEW OF STATE VECTOR */
/* TODO: CHANGE BLOCK_VECTOR TO GRAPH_VECTOR (DONE) */
/* TODO: MAKE GRAPH_VECTORS VIEWS OF STATE (DONE) */

ok_status pogs_solve(pogs_solver *solver, const function_vector *f,
	const function_vector *g, const pogs_settings *settings,
	pogs_info *info, pogs_output *output)
{
	if (!solver || !settings || !info || !output)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_FNVECTOR(g);

	ok_status err = OPTKIT_SUCCESS;

	/* SETUP */
	OK_TIMER t = ok_timer_tic();

	if (settings->diagnostic)
		OK_CHECK_ERR( err, pogs_setup_diagnostics(solver,
			settings->maxiter) );


	/* TODO: accel init here or elsewhere? */
	if (settings->accelerate)
		OK_CHECK_ERR( err, ACCELERATOR(accelerator_init)(solver->aa,
			solver->z->fixed_point_iterate->size,
			(size_t) settings->anderson_lookback, (size_t) 2) );


	OK_CHECK_ERR( err, pogs_update_settings(solver->settings, settings) );
	OK_CHECK_ERR( err, pogs_scale_objectives(solver->f, solver->g,
		solver->W->d, solver->W->e, f, g) );
	if (settings->warmstart)
		OK_CHECK_ERR( err, pogs_set_z0(solver) );

	// solver->aa->mu_regularization = settings->anderson_regularization;
	if ((settings->warmstart || settings->resume) && settings->accelerate){
		OK_CHECK_ERR( err, ACCELERATOR(set_x0)(
			solver->aa, solver->z->fixed_point_iterate) );
		OK_CHECK_ERR( err, vector_set_all(solver->z->state, kZero) );
	}

	if (!(settings->resume))
		solver->rho = settings->rho;

	info->setup_time = ok_timer_toc(t);
	if (!(settings->warmstart || settings->resume))
		info->setup_time += solver->init_time;

	/* SOLVE */
	t = ok_timer_tic();
	OK_CHECK_ERR( err, pogs_solver_loop(solver, info) );
	info->solve_time = ok_timer_toc(t);

	/* TODO: move with accel init, if needed */
	if (settings->accelerate)
		OK_MAX_ERR( err, ACCELERATOR(accelerator_free)(solver->aa) );

	/* UNSCALE */
	OK_CHECK_ERR( err, pogs_unscale_output(output, solver->z, solver->W->d,
		solver->W->e, solver->rho, solver->settings->suppress) );

	if (settings->diagnostic)
		OK_CHECK_ERR( err, pogs_emit_diagnostics(output, solver) );

	return err;
}

ok_status pogs_finish(pogs_solver *solver, const int reset)
{
	ok_status err = OK_SCAN_ERR( pogs_solver_free(solver) );
	ok_free(solver);
	if (reset)
		OK_MAX_ERR( err, ok_device_reset() );
	return err;
}

ok_status pogs(pogs_solver_data *A, const pogs_solver_flags *flags,
	const function_vector *f, const function_vector *g,
	const pogs_settings *settings, pogs_info *info, pogs_output *output,
	const int reset)
{
	ok_status err = OPTKIT_SUCCESS;
	pogs_solver *solver = pogs_init(A, flags);
	if (!solver || !solver->W)
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_ERR( err, pogs_solve(solver, f, g, settings, info, output) );
	OK_MAX_ERR( err, pogs_finish(solver, reset) );
	return err;
}

ok_status pogs_export_solver(pogs_solver_private_data *priv_data, ok_float *state,
	ok_float *rho, pogs_solver_flags *flags, const pogs_solver *solver)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!solver)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_ERR( err, POGS(save_work)(priv_data, flags, solver->W) );
	OK_CHECK_ERR( err, pogs_solver_save_state(state, rho, solver) );
	return err;
}

pogs_solver * pogs_load_solver(const pogs_solver_private_data *data,
	const ok_float *state, const ok_float rho,
	const pogs_solver_flags *flags)
{
	ok_status err;
	pogs_solver *solver = OK_NULL;
	pogs_solver_data *A = OK_NULL;

	err = OK_SCAN_ERR( POGS(get_init_data)(&A, data, flags) );
	ok_alloc(solver, sizeof(*solver));
	OK_CHECK_ERR( err, pogs_solver_alloc(solver, A, flags) );
	OK_CHECK_ERR( err, POGS(load_work)(solver->W, data, flags) );
	OK_CHECK_ERR( err, pogs_solver_load_state(solver, state, rho) );
	if (err) {
		OK_MAX_ERR( err, pogs_solver_free(solver) );
		ok_free(solver);
	}
	return solver;
}

ok_status pogs_solver_save_work(pogs_solver_private_data *priv_data,
	pogs_solver_flags *flags, const pogs_solver *solver)
{
	return OK_SCAN_ERR( POGS(save_work)(priv_data, flags,
		solver->W) );
}

ok_status pogs_solver_load_work(pogs_solver *solver,
	const pogs_solver_private_data *priv_data,
	const pogs_solver_flags *flags)
{
	return OK_SCAN_ERR( POGS(load_work)(solver->W, priv_data,
		flags) );
}

ok_status pogs_solver_save_state(ok_float *state, ok_float *rho,
	const pogs_solver *solver)
{
	if (!solver || !solver->z || !solver->z->state || !state || !rho)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_RETURNIF_ERR( vector_memcpy_av(state, solver->z->state, 1) );
	*rho = solver->rho;
	return OPTKIT_SUCCESS;
}
ok_status pogs_solver_load_state(pogs_solver *solver, const ok_float *state,
	const ok_float rho)
{
	if (!solver || !solver->z || !solver->z->state  || !state)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_RETURNIF_ERR( vector_memcpy_va(solver->z->state, state, 1) );
	solver->rho = rho;
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
