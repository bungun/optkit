#include "optkit_abstract_pogs.h"

#ifdef __cplusplus
extern "C" {
#endif

const ok_float kProjectorTolInitial = (ok_float) 1e-6;

POGS_PRIVATE ok_status pogs_work_alloc(pogs_work ** W, operator * A, int direct)
{
	int dense_or_sparse = (A->kind == OkOperatorDense ||
		A->kind == OkOperatorSparseCSC ||
		A->kind == OkOperatorSparseCSR);
	pogs_work * W_ = OK_NULL;
	W_ = malloc(sizeof(*W_));
	W_->A = A;

	/* set projector */
	if (direct && A->kind == OkOperatorDense)
		W_->P = dense_direct_projector_alloc(
				dense_operator_get_matrix_pointer(W_->A));
	else
		W_->P = indirect_projector_generic_alloc(W_->A);

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

	W_->d = malloc(sizeof(*(W_->d)));
	W_->e = malloc(sizeof(*(W_->e)));
	vector_calloc(W_->d, A->size1);
	vector_calloc(W_->e, A->size2);
	W_->skinny = (A->size1 >= A->size2);
	W_->normalized = 0;
	W_->equilibrated = 0;
	* W = W_;

	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status pogs_work_free(pogs_work * W)
{
	W->P->free(W->P->data);
	ok_free(W->P);
	vector_free(W->d);
	vector_free(W->e);
	ok_free(W);
	return OPTKIT_SUCCESS;
}


POGS_PRIVATE ok_status block_vector_alloc(block_vector ** z, size_t m, size_t n)
{
	block_vector * z_ = OK_NULL;
	z_ = malloc(sizeof(*z_));
	z_->size = m + n;
	z_->m = m;
	z_->n = n;
	z_->vec = malloc(sizeof(*(z_->vec)));
	z_->x = malloc(sizeof(*(z_->x)));
	z_->y = malloc(sizeof(*(z_->y)));
	vector_calloc(z_->vec, m + n);
	vector_subvector(z_->y, z_->vec, 0, m);
	vector_subvector(z_->x, z_->vec, m, n);
	* z = z_;
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status block_vector_free(block_vector * z)
{
	vector_free(z->vec);
	ok_free(z->x);
	ok_free(z->y);
	ok_free(z->vec);
	ok_free(z);
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status pogs_variables_alloc(pogs_variables ** z, size_t m,
	size_t n)
{
	pogs_variables * z_ = OK_NULL;
	z_ = malloc(sizeof(*z_));
	z_->m = m;
	z_->n = n;
	block_vector_alloc(&(z_->primal), m, n);
	block_vector_alloc(&(z_->primal12), m, n);
	block_vector_alloc(&(z_->dual), m ,n);
	block_vector_alloc(&(z_->dual12), m, n);
	block_vector_alloc(&(z_->prev), m ,n);
	block_vector_alloc(&(z_->temp), m, n);
	*z = z_;
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status pogs_variables_free(pogs_variables * z)
{
	block_vector_free(z->primal);
	block_vector_free(z->primal12);
	block_vector_free(z->dual);
	block_vector_free(z->dual12);
	block_vector_free(z->prev);
	block_vector_free(z->temp);
	ok_free(z);
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status pogs_solver_alloc(pogs_solver ** solver, operator * A,
	int direct)
{
	pogs_solver * s = OK_NULL;
	s = malloc(sizeof(*s));
	s->settings = malloc(sizeof(*(s->settings)));
	set_default_settings(s->settings);
	s->f = malloc(sizeof(*(s->f)));
	s->g = malloc(sizeof(*(s->g)));
	s->f->objectives = OK_NULL;
	s->g->objectives = OK_NULL;
	function_vector_calloc(s->f, A->size1);
	function_vector_calloc(s->g, A->size2);
	pogs_variables_alloc(&(s->z), A->size1, A->size2);
	pogs_work_alloc(&(s->W), A, direct);
	blas_make_handle(&(s->linalg_handle));
	s->rho = kOne;
	*solver = s;
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status pogs_solver_free(pogs_solver * solver)
{
	if (solver) {
		blas_destroy_handle(solver->linalg_handle);
		pogs_work_free(solver->W);
		pogs_variables_free(solver->z);
		ok_free(solver->settings);
		function_vector_free(solver->f);
		function_vector_free(solver->g);
		ok_free(solver->f);
		ok_free(solver->g);
		ok_free(solver);
	}
	return OPTKIT_SUCCESS;
}


POGS_PRIVATE ok_status update_settings(pogs_settings * settings,
	const pogs_settings * input)
{
	memcpy((void *) settings, (void *) input, sizeof(*settings));
	return OPTKIT_SUCCESS;
}


POGS_PRIVATE ok_status equilibrate(void * linalg_handle, pogs_work * W,
	const ok_float pnorm)
{
	ok_status res = OPTKIT_SUCCESS;
	res = W->operator_equilibrate(linalg_handle, W->A, W->d, W->e, pnorm);
	W->equilibrated = (res == OPTKIT_SUCCESS);
	return res;
}

POGS_PRIVATE ok_float estimate_norm(void * linalg_handle, pogs_work * W)
{
	return operator_estimate_norm(linalg_handle, W->A);
}

POGS_PRIVATE ok_status normalize_DAE(void * linalg_handle, pogs_work * W)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t m = W->A->size1,  n = W->A->size2;
	size_t mindim = m < n ? m : n;
	ok_float sqrt_mindim = MATH(sqrt)((ok_float) mindim), factor;

	W->normalized = projector_normalization(W->P);
	W->normA = projector_get_norm(W->P);

	if (!(W->normalized)) {
		W->normA = estimate_norm(linalg_handle, W) / sqrt_mindim;
 		err = W->operator_scale(W->A, kOne / W->normA);
		W->normalized = 1;
	}
	factor = MATH(sqrt)(MATH(sqrt)(
		((ok_float) n * blas_dot(linalg_handle, W->d, W->d)) /
		((ok_float) m * blas_dot(linalg_handle, W->e, W->e))  ));
	vector_scale(W->d, kOne / (MATH(sqrt)(W->normA) * factor));
	vector_scale(W->e, factor / MATH(sqrt)(W->normA));

	return err;
}

POGS_PRIVATE ok_status update_problem(pogs_solver * solver, FunctionVector * f,
	FunctionVector * g)
{
	function_vector_memcpy_va(solver->f, f->objectives);
	function_vector_memcpy_va(solver->g, g->objectives);
	function_vector_div(solver->f, solver->W->d);
	function_vector_mul(solver->g, solver->W->e);
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status initialize_variables(pogs_solver * solver)
{
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


POGS_PRIVATE pogs_tolerances make_tolerances(const pogs_settings * settings,
	size_t m, size_t n)
{
	return (pogs_tolerances){
		.primal = kZero,
		.dual = kZero,
		.gap = kZero,
		.reltol = settings->reltol,
		.abstol = settings->abstol,
		.atolm = MATH(sqrt)( (ok_float) m ) * settings->abstol,
		.atoln = MATH(sqrt)( (ok_float) n ) * settings->abstol,
		.atolmn = MATH(sqrt)( (ok_float)(m * n) ) * settings->abstol
	};
}

POGS_PRIVATE ok_status update_objective(void * linalg_handle, FunctionVector * f,
	FunctionVector * g, ok_float rho, pogs_variables * z,
	pogs_objectives * obj)
{
	obj->gap = MATH(fabs)(blas_dot(linalg_handle, z->primal12->vec,
		z->dual12->vec));
	obj->primal = FuncEvalVector(f, z->primal12->y) +
			FuncEvalVector(g, z->primal12->x);
	obj->dual = obj->primal - obj->gap;
	return OPTKIT_SUCCESS;
}

/*
 * eps_primal = eps_abs + eps_rel * sqrt(m) * ||y^k+1/2||
 * eps_dual = eps_abs + eps_rel * sqrt(n) * ||xt^k+1/2||
 * eps_gap = eps_abs + eps_rel * sqrt(mn) * ||z^k|| * ||z^k+1/2||
 */
POGS_PRIVATE ok_status update_tolerances(void * linalg_handle, pogs_variables * z,
	pogs_objectives * obj, pogs_tolerances * eps)
{
	eps->primal = eps->atolm + eps->reltol * MATH(sqrt)(blas_dot(
		linalg_handle, z->primal12->y, z->primal12->y));
	eps->dual = eps->atoln + eps->reltol * MATH(sqrt)(blas_dot(
		linalg_handle, z->dual12->x, z->dual12->x));
	eps->gap = eps->atolmn + eps->reltol * MATH(sqrt)(blas_dot(
		linalg_handle, z->primal->vec,
		z->primal->vec)) * MATH(sqrt)(blas_dot(linalg_handle,
		z->primal12->vec, z->primal12->vec));
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

	pogs_variables * z = solver->z;
	operator * A = solver->W->A;

	res->gap = obj->gap;

	vector_memcpy_vv(z->temp->y, z->primal12->y);
	A->fused_apply(A->data, kOne, z->primal12->x, -kOne, z->temp->y);
	res->primal = MATH(sqrt)(blas_dot(linalg_handle, z->temp->y,
		z->temp->y));

	vector_memcpy_vv(z->temp->x, z->dual12->x);
	A->fused_adjoint(A->data, kOne, z->dual12->y, kOne, z->temp->x);
	res->dual = MATH(sqrt)(blas_dot(linalg_handle, z->temp->x, z->temp->x));

	return OPTKIT_SUCCESS;
}

POGS_PRIVATE int check_convergence(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps,
	int gapstop)
{
	update_objective(linalg_handle, solver->f, solver->g,
		solver->rho, solver->z, obj);
	update_tolerances(linalg_handle, solver->z, obj, eps);
	update_residuals(linalg_handle, solver, obj, res, eps);

	return (res->primal < eps->primal) && (res->dual < eps->dual) &&
		(res->gap < eps->gap || !(gapstop));
}


/* z^k <- z^{k+1} */
POGS_PRIVATE ok_status set_prev(pogs_variables * z)
{
	vector_memcpy_vv(z->prev->vec, z->primal->vec);
	return OPTKIT_SUCCESS;
}

/*
 * update z^{k + 1/2} according to following rule:
 *
 *	y^{k+1/2} = Prox_{rho, f} (y^k - yt^k)
 *	x^{k+1/2} = Prox_{rho, g} (x^k - xt^k)
 */
POGS_PRIVATE ok_status prox(void * linalg_handle, FunctionVector * f,
	FunctionVector * g, pogs_variables * z, ok_float rho)
{
	vector_memcpy_vv(z->temp->vec, z->primal->vec);
	blas_axpy(linalg_handle, -kOne, z->dual->vec, z->temp->vec);
	ProxEvalVector(f, rho, z->temp->y, z->primal12->y);
	ProxEvalVector(g, rho, z->temp->x, z->primal12->x);
	return OPTKIT_SUCCESS;
}

/*
 * update z^{k+1} according to rule:
 *	( x^{k+1}, y^{k+1} ) =
 *		Proj_{y=Ax} (x^{k+1/2 + xt^k, y^{k+1/2} + yt^k)
 */
POGS_PRIVATE ok_status project_primal(void * linalg_handle, projector * proj,
	pogs_variables * z, ok_float alpha, ok_float tol)
{
	vector_set_all(z->temp->vec, kZero);
	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->temp->vec);
	proj->project(proj->data, z->temp->x, z->temp->y, z->primal->x,
		z->primal->y, tol);

	return OPTKIT_SUCCESS;
}

/*
 * update zt^{k+1/2} and zt^{k+1} according to:
 *
 * 	zt^{k+1/2} = z^{k+1/2} - z^k + zt^k
 * 	zt^{k+1}   = zt^k + z^{k+1/2} - z^k+1
 */
POGS_PRIVATE ok_status update_dual(void * linalg_handle, pogs_variables * z,
	ok_float alpha)
{
	vector_memcpy_vv(z->dual12->vec, z->primal12->vec);
	blas_axpy(linalg_handle, -kOne, z->prev->vec, z->dual12->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->dual12->vec);

	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->dual->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->dual->vec);
	blas_axpy(linalg_handle, -kOne, z->primal->vec, z->dual->vec);

	return OPTKIT_SUCCESS;
}


/*
 * change solver->rho to balance primal and dual convergence
 * (and rescale z->dual accordingly)
 */
POGS_PRIVATE ok_status adaptrho(pogs_solver * solver, adapt_params * params,
	pogs_residuals * res, pogs_tolerances * eps, uint k)
{
	if (!(solver->settings->adaptiverho)) return OPTKIT_SUCCESS;

	if (res->dual < params->xi * eps->dual &&
		res->primal > params->xi * eps->primal &&
		kTAU * (ok_float) k > params->l) {

		if (solver->rho < kRHOMAX) {
		  solver->rho *= params->delta;
		  if (solver->settings->verbose > 2)
			  printf("+RHO: %.3e\n", solver->rho);
		  vector_scale(solver->z->dual->vec, kOne / params->delta);
		  params->delta = (params->delta * kGAMMA < kDELTAMAX) ?
		  		  params->delta * kGAMMA : kDELTAMAX;
		  params->u = (ok_float) k;
		}
	} else if (res->dual > params->xi * eps->dual &&
			res->primal < params->xi * eps->primal &&
			kTAU * (ok_float) k > (ok_float) params->u) {

		if (solver->rho > kRHOMIN) {
		  solver->rho /= params->delta;
		  if (solver->settings->verbose > 2)
			  printf("-RHO: %.3e\n", solver->rho);
		  vector_scale(solver->z->dual->vec, params->delta);
		  params->delta = (params->delta * kGAMMA < kDELTAMAX) ?
		  		  params->delta * kGAMMA : kDELTAMAX;
		  params->l = (ok_float) k;
		}
	} else if (res->dual < params->xi * eps->dual &&
			res->primal < params->xi * eps->primal) {
		params->xi *= kKAPPA;
	} else {
		params->delta = (params->delta / kGAMMA > kDELTAMIN) ?
				params->delta / kGAMMA : kDELTAMIN;
	}

	return OPTKIT_SUCCESS;
}

/*
 * copy pogs variables to outputs:
 *
 * 	x^{k + 1/2} --> output.x
 * 	y^{k + 1/2} --> output.y
 * 	-rho * xt^{k + 1/2} --> output.mu
 * 	-rho * yt^{k + 1/2} --> output.nu
 *
 *
 * output suppression levels:
 *
 *	0: copy (x, y, mu, nu), suppress ()
 *	1: copy (x, y, nu), suppress (mu)
 *	2: copy (x, nu), suppress (y, mu)
 *	3: copy (x), suppress (y, mu, nu)
 */
POGS_PRIVATE ok_status copy_output(pogs_solver * solver, pogs_output * output)
{
	vector * d = solver->W->d;
	vector * e = solver->W->e;
	pogs_variables * z = solver->z;
	uint suppress = solver->settings->suppress;

	vector_memcpy_vv(z->temp->vec, z->primal12->vec);
	vector_mul(z->temp->x, e);
	vector_memcpy_av(output->x, z->temp->x, 1);

	if (suppress < 2) {
		vector_div(z->temp->y, d);
		vector_memcpy_av(output->y, z->temp->y, 1);
	}


	if (suppress < 3) {
		vector_memcpy_vv(z->temp->vec, z->dual12->vec);
		vector_scale(z->temp->vec, -solver->rho);
		vector_mul(z->temp->y, d);
		vector_memcpy_av(output->nu, z->temp->y, 1);

		if (suppress < 1) {
			vector_div(z->temp->x, e);
			vector_memcpy_av(output->mu, z->temp->x, 1);
		}
	}

	return OPTKIT_SUCCESS;
}

POGS_PRIVATE void print_header_string()
{
	printf("\n   #    %s    %s    %s   %s   %s        %s    %s\n",
		"res_pri", "eps_pri", "res_dual", "eps_dual",
		"gap", "eps_gap", "objective");
	printf("   -    %s    %s    %s   %s   %s        %s    %s\n",
		"-------", "-------", "--------", "--------",
		"---", "-------", "---------");
}

POGS_PRIVATE void print_iter_string(pogs_residuals * res, pogs_tolerances * eps,
	pogs_objectives * obj, uint k)
{
	printf("   %u: %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e\n",
		k, res->primal, eps->primal, res->dual, eps->dual,
		res->gap, eps->gap, obj->primal);
}

POGS_PRIVATE ok_status pogs_solver_loop(pogs_solver * solver, pogs_info * info)
{
	/* declare / get handles to all auxiliary types */
	int err = 0, converged = 0;
	uint k, PRINT_ITER = 10000u;
	adapt_params rho_params = (adapt_params){kDELTAMIN, kZero, kZero, kOne};
	pogs_settings * settings = solver->settings;
	pogs_variables * z = solver->z;
	pogs_objectives obj = (pogs_objectives){OK_NAN, OK_NAN, OK_NAN};
	pogs_residuals res = (pogs_residuals){OK_NAN, OK_NAN, OK_NAN};
	pogs_tolerances eps = make_tolerances(settings,
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
	if (!err && settings->verbose > 0)
		print_header_string();

	/* iterate until converged, or error/maxiter reached */
	for (k = 1; !err && k <= settings->maxiter; ++k) {
		set_prev(z);
		prox(linalg_handle, solver->f, solver->g, z, solver->rho);
		project_primal(linalg_handle, solver->W->P, z,
			settings->alpha, tol_proj);
		update_dual(linalg_handle, z, settings->alpha);

		converged = check_convergence(linalg_handle, solver,
			&obj, &res, &eps, settings->gapstop);

		if (settings->verbose > 0 && (k % PRINT_ITER == 0 ||
					      converged ||
			 		      k == settings->maxiter)
		   )
			print_iter_string(&res, &eps, &obj, k);

		if (converged || k == settings->maxiter)
			break;

		if (settings->adaptiverho)
			adaptrho(solver, &rho_params, &res, &eps, k);
	}

	if (!converged && k == settings->maxiter)
		printf("reached max iter = %u\n", k);

	/* update info */
	info->rho = solver->rho;
	info->obj = obj.primal;
	info->converged = converged;
	info->err = err;
	info->k = k;

	return OPTKIT_SUCCESS;
}

void set_default_settings(pogs_settings * s)
{
	s->alpha = kALPHA;
	s->rho = kOne;
	s->abstol = kATOL;
	s->reltol = kRTOL;
	s->maxiter = kMAXITER;
	s->verbose = kVERBOSE;
	s->suppress = kSUPPRESS;
	s->adaptiverho = kADAPTIVE;
	s->gapstop = kGAPSTOP;
	s->warmstart = kWARMSTART;
	s->resume = kRESUME;
	s->x0 = OK_NULL;
	s->nu0 = OK_NULL;
}


pogs_solver * pogs_init(operator * A, const int direct,
	const ok_float equil_norm)
{
	ok_status err;
	int normalize;
	pogs_solver * solver = OK_NULL;
	OK_TIMER t = tic();

	/* make solver variables */
	pogs_solver_alloc(&solver, A, direct);

	/* equilibrate A as (D * A_equil * E) = A */
	err = equilibrate(solver->linalg_handle, solver->W, equil_norm);

	if (!err) {
		/* make projector; normalize A; adjust d, e accordingly */
		normalize = (int) (solver->W->P->kind == OkProjectorDenseDirect);
		solver->W->P->initialize(solver->W->P->data, normalize);
		err = normalize_DAE(solver->linalg_handle, solver->W);
		solver->init_time = toc(t);
	}

	if (err) {
		pogs_solver_free(solver);
		solver = OK_NULL;
	}

	return solver;
}


ok_status pogs_solve(pogs_solver * solver, FunctionVector * f,
	FunctionVector * g, const pogs_settings * settings, pogs_info * info,
	pogs_output * output)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!solver)
		return OPTKIT_ERROR_UNALLOCATED;

	OK_TIMER t = tic();

	/* copy settings */
	OPTKIT_RETURN_IF_ERROR(update_settings(solver->settings, settings));

	/* copy and scale function vectors */
	OPTKIT_RETURN_IF_ERROR(update_problem(solver, f, g));

	/* get warm start variables */
	if (settings->warmstart)
		OPTKIT_RETURN_IF_ERROR(initialize_variables(solver));
	if ( !(settings->resume) )
		solver->rho = settings->rho;

	info->setup_time = toc(t);
	if (!(settings->warmstart || settings->resume))
		info->setup_time += solver->init_time;

	/* run solver */
	t = tic();
	err = pogs_solver_loop(solver, info);
	info->solve_time = toc(t);

	/* unscale output */
	OPTKIT_CHECK_ERROR(&err, copy_output(solver, output));

	return err;
}

ok_status pogs_finish(pogs_solver * solver, const int reset)
{
	ok_status err = OPTKIT_SUCCESS;
	if (solver)
		OPTKIT_CHECK_ERROR(&err, pogs_solver_free(solver));
	if (reset)
		OPTKIT_CHECK_ERROR(&err, ok_device_reset());
	return err;
}

ok_status pogs(operator * A, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	const int direct, const ok_float equil_norm, const int reset)
{
	ok_status err = OPTKIT_SUCCESS;
	pogs_solver * solver = OK_NULL;
	solver = pogs_init(A, direct, equil_norm);
	OPTKIT_CHECK_ERROR(&err,
		pogs_solve(solver, f, g, settings, info, output));
	OPTKIT_CHECK_ERROR(&err, pogs_finish(solver, reset));
	return err;
}


operator * pogs_dense_operator_gen(const ok_float * A, size_t m, size_t n,
	enum CBLAS_ORDER order)
{
	matrix * A_mat = OK_NULL;
	A_mat = malloc(sizeof(*A_mat));
	matrix_calloc(A_mat, m, n, order);
	matrix_memcpy_ma(A_mat, A, order);
	return dense_operator_alloc(A_mat);
}

operator * pogs_sparse_operator_gen(const ok_float * val, const ok_int * ind,
	const ok_int * ptr, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order)
{
	void * handle = OK_NULL;
	sp_matrix * A = OK_NULL;
	A = malloc(sizeof(*A));
	sp_matrix_calloc(A, m, n, nnz, order);
	sp_make_handle(&handle);
	sp_matrix_memcpy_ma(handle, A, val, ind, ptr);
	sp_destroy_handle(handle);
	return sparse_operator_alloc(A);
}

void pogs_dense_operator_free(operator * A)
{
	matrix * A_mat = dense_operator_get_matrix_pointer(A);
	A->free(A->data);
	matrix_free(A_mat);
	ok_free(A_mat);
}

void pogs_sparse_operator_free(operator * A)
{
	sp_matrix * A_mat = sparse_operator_get_matrix_pointer(A);
	A->free(A->data);
	sp_matrix_free(A_mat);
	ok_free(A_mat);
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
