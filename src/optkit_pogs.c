#include "optkit_pogs.h"

#ifdef __cplusplus
extern "C" {
#endif

int private_api_accessible()
{
#ifdef OK_DEBUG_PYTHON
	return 1;
#else
	return 0;
#endif
}

int is_direct()
{
	return sizeof(projector_) == sizeof(direct_projector);
}

POGS_PRIVATE void pogs_matrix_alloc(pogs_matrix ** M, size_t m, size_t n,
	enum CBLAS_ORDER ord)
{
	pogs_matrix * M_ = OK_NULL;
	M_ = malloc(sizeof(*M_));
	M_->A = malloc(sizeof(matrix));
	matrix_calloc(M_->A, m, n, ord);
	M_->P = malloc(sizeof(projector_));
	PROJECTOR(alloc(M_->P, M_->A));
	M_->d = malloc(sizeof(vector));
	M_->e = malloc(sizeof(vector));
	vector_calloc(M_->d, m);
	vector_calloc(M_->e, n);
	M_->skinny = (m >= n);
	M_->normalized = 0;
	M_->equilibrated = 0;
	*M = M_;
}

POGS_PRIVATE void pogs_matrix_free(pogs_matrix * M)
{
	PROJECTOR(free(M->P));
	ok_free(M->P);
	matrix_free(M->A);
	vector_free(M->d);
	vector_free(M->e);
	ok_free(M);
}


POGS_PRIVATE void block_vector_alloc(block_vector ** z, size_t m, size_t n)
{
	block_vector * z_ = OK_NULL;
	z_ = malloc(sizeof(*z_));
	z_->size = m + n;
	z_->m = m;
	z_->n = n;
	z_->vec = malloc(sizeof(vector));
	z_->x = malloc(sizeof(vector));
	z_->y = malloc(sizeof(vector));
	vector_calloc(z_->vec, m + n);
	vector_subvector(z_->y, z_->vec, 0, m);
	vector_subvector(z_->x, z_->vec, m, n);
	* z = z_;
}

POGS_PRIVATE void block_vector_free(block_vector * z)
{
	vector_free(z->vec);
	ok_free(z->x);
	ok_free(z->y);
	ok_free(z->vec);
	ok_free(z);
}

POGS_PRIVATE void pogs_variables_alloc(pogs_variables ** z, size_t m, size_t n)
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
}

POGS_PRIVATE void pogs_variables_free(pogs_variables * z)
{
	block_vector_free(z->primal);
	block_vector_free(z->primal12);
	block_vector_free(z->dual);
	block_vector_free(z->dual12);
	block_vector_free(z->prev);
	block_vector_free(z->temp);
	ok_free(z);
}

POGS_PRIVATE void pogs_solver_alloc(pogs_solver ** solver, size_t m, size_t n,
	  enum CBLAS_ORDER ord)
{
	pogs_solver * s = OK_NULL;
	s = malloc(sizeof(*s));
	s->settings = malloc(sizeof(*(s->settings)));
	set_default_settings(s->settings);
	s->f = malloc(sizeof(*(s->f)));
	s->g = malloc(sizeof(*(s->g)));
	s->f->objectives = OK_NULL;
	s->g->objectives = OK_NULL;
	function_vector_calloc(s->f, m);
	function_vector_calloc(s->g, n);
	pogs_variables_alloc(&(s->z), m, n);
	pogs_matrix_alloc(&(s->M), m, n, ord);
	blas_make_handle(&(s->linalg_handle));
	s->rho = kOne;
	* solver = s;
}

POGS_PRIVATE void pogs_solver_free(pogs_solver * solver)
{
	blas_destroy_handle(solver->linalg_handle);
	pogs_matrix_free(solver->M);
	pogs_variables_free(solver->z);
	ok_free(solver->settings);
	function_vector_free(solver->f);
	function_vector_free(solver->g);
	ok_free(solver->f);
	ok_free(solver->g);
	ok_free(solver);
}


POGS_PRIVATE void update_settings(pogs_settings * settings,
	const pogs_settings * input)
{
	memcpy( (void *) settings,  (void *) input, sizeof(pogs_settings));
}


POGS_PRIVATE void equilibrate(void * linalg_handle, ok_float * A_orig,
	pogs_matrix * M, EQUILIBRATOR equil, enum CBLAS_ORDER ord)
{
	if (equil == EquilSinkhorn)
		sinkhorn_knopp(linalg_handle, A_orig, M->A, M->d, M->e, ord);
	else
		dense_l2(linalg_handle, A_orig, M->A, M->d, M->e, ord);
	M->equilibrated = 1;
}

/* STUB */
POGS_PRIVATE ok_float estimate_norm(void * linalg_handle, pogs_matrix * M)
{
	return kOne;
}

POGS_PRIVATE void normalize_DAE(void * linalg_handle, pogs_matrix * M)
{
	size_t m = M->A->size1,  n = M->A->size2;
	size_t mindim = m < n ? m : n;
	ok_float factor;

	M->normalized = M->P->normalized;
	M->normA = M->P->normA;


	if (!(M->normalized)) {
		M->normA = estimate_norm(linalg_handle, M) /
			MATH(sqrt)((ok_float) mindim);
		matrix_scale(M->A, kOne / M->normA);
		M->normalized = 1;
	}
	factor = MATH(sqrt)(MATH(sqrt)(
		((ok_float) n * blas_dot(linalg_handle, M->d, M->d)) /
		((ok_float) m * blas_dot(linalg_handle, M->e, M->e))  ));
	vector_scale(M->d, kOne / (MATH(sqrt)(M->normA) * factor));
	vector_scale(M->e, factor / MATH(sqrt)(M->normA));
}


POGS_PRIVATE void update_problem(pogs_solver * solver, FunctionVector * f,
	FunctionVector * g)
{
	function_vector_memcpy_va(solver->f, f->objectives);
	function_vector_memcpy_va(solver->g, g->objectives);
	function_vector_div(solver->f, solver->M->d);
	function_vector_mul(solver->g, solver->M->e);
}

POGS_PRIVATE void initialize_variables(pogs_solver * solver)
{
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
		.atolm = MATH(sqrt)((ok_float) m) * settings->abstol,
		.atoln = MATH(sqrt)((ok_float) n) * settings->abstol,
		.atolmn = MATH(sqrt)((ok_float) (m * n)) * settings->abstol,
	};
}

POGS_PRIVATE void update_objective(void * linalg_handle, FunctionVector * f,
	FunctionVector * g, ok_float rho, pogs_variables * z,
	pogs_objectives * obj)
{
	obj->gap = MATH(fabs)(blas_dot(linalg_handle, z->primal12->vec,
		z->dual12->vec));
	obj->primal = FuncEvalVector(f, z->primal12->y) +
			FuncEvalVector(g, z->primal12->x);
	obj->dual = obj->primal - obj->gap;
}

/*
 * eps_primal = eps_abs + eps_rel * sqrt(m) * ||y^k+1/2||
 * eps_dual = eps_abs + eps_rel * sqrt(n) * ||xt^k+1/2||
 * eps_gap = eps_abs + eps_rel * sqrt(mn) * ||z^k|| * ||z^k+1/2||
 */
POGS_PRIVATE void update_tolerances(void * linalg_handle, pogs_variables * z,
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
}


/*
 * residual for primal feasibility
 *	||Ax^(k+1/2) - y^(k+1/2)||
 *
 * residual for dual feasibility
 * 	||A'yt^(k+1/2) - xt^(k+1/2)||
 */
POGS_PRIVATE void update_residuals(void * linalg_handle, pogs_solver * solver,
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps)
{

	pogs_variables * z = solver->z;
	matrix * A = solver->M->A;

	res->gap = obj->gap;

	vector_memcpy_vv(z->temp->y, z->primal12->y);
	blas_gemv(linalg_handle, CblasNoTrans, kOne,
		A, z->primal12->x, -kOne, z->temp->y);
	res->primal = MATH(sqrt)(blas_dot(linalg_handle,
		z->temp->y, z->temp->y));

	vector_memcpy_vv(z->temp->x, z->dual12->x);
	blas_gemv(linalg_handle, CblasTrans, kOne,
		A, z->dual12->y, kOne, z->temp->x);
	res->dual = MATH(sqrt)(blas_dot(linalg_handle,
		z->temp->x, z->temp->x));
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
POGS_PRIVATE void set_prev(pogs_variables * z)
{
	vector_memcpy_vv(z->prev->vec, z->primal->vec);
}

/*
 * update z^{k + 1/2} according to following rule:
 *
 *	y^{k+1/2} = Prox_{rho, f} (y^k - yt^k)
 *	x^{k+1/2} = Prox_{rho, g} (x^k - xt^k)
 */
POGS_PRIVATE void prox(void * linalg_handle, FunctionVector * f,
	FunctionVector * g, pogs_variables * z, ok_float rho)
{
	vector_memcpy_vv(z->temp->vec, z->primal->vec);
	blas_axpy(linalg_handle, -kOne, z->dual->vec, z->temp->vec);
	ProxEvalVector(f, rho, z->temp->y, z->primal12->y);
	ProxEvalVector(g, rho, z->temp->x, z->primal12->x);
}

/*
 * update z^{k+1} according to rule:
 *	( x^{k+1}, y^{k+1} ) =
 *		Proj_{y=Ax} (x^{k+1/2 + xt^k, y^{k+1/2} + yt^k)
 */
POGS_PRIVATE void project_primal(void * linalg_handle, projector_ * proj,
	pogs_variables * z,  ok_float alpha)
{
	vector_set_all(z->temp->vec, kZero);
	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->temp->vec);
	PROJECTOR(project)(linalg_handle, proj, z->temp->x, z->temp->y,
		z->primal->x, z->primal->y);
}


/*
 * update zt^{k+1/2} and zt^{k+1} according to:
 *
 * 	zt^{k+1/2} = z^{k+1/2} - z^k + zt^k
 * 	zt^{k+1}   = zt^k + z^{k+1/2} - z^k+1
 */
POGS_PRIVATE void update_dual(void * linalg_handle, pogs_variables * z,
	ok_float alpha)
{
	vector_memcpy_vv(z->dual12->vec, z->primal12->vec);
	blas_axpy(linalg_handle, -kOne, z->prev->vec, z->dual12->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->dual12->vec);

	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->dual->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->dual->vec);
	blas_axpy(linalg_handle, -kOne, z->primal->vec, z->dual->vec);
}


/*
 * change solver->rho to balance primal and dual convergence
 * (and rescale z->dual accordingly)
 */
POGS_PRIVATE void adaptrho(pogs_solver * solver, adapt_params * params,
	pogs_residuals * res, pogs_tolerances * eps, uint k)
{
	if (!(solver->settings->adaptiverho)) return;

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
POGS_PRIVATE void copy_output(pogs_solver * solver, pogs_output * output)
{
	vector * d = solver->M->d;
	vector * e = solver->M->e;
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

POGS_PRIVATE void pogs_solver_loop(pogs_solver * solver, pogs_info * info)
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
		project_primal(linalg_handle, solver->M->P, z,
			settings->alpha);
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


pogs_solver * pogs_init(ok_float * A, size_t m, size_t n, enum CBLAS_ORDER ord,
	EQUILIBRATOR equil)
{
	pogs_solver * solver = OK_NULL;
	OK_TIMER t = tic();

	/* make variables, matrix */
	pogs_solver_alloc(&solver, m , n, ord);

	/* equilibrate A as (D * A_equil * E) = A */
	equilibrate(solver->linalg_handle, A, solver->M, equil, ord);

	/* make projector; normalize A; adjust d, e accordingly */
	PROJECTOR(initialize)(solver->linalg_handle, solver->M->P, 1);
	normalize_DAE(solver->linalg_handle, solver->M);

	solver->init_time = toc(t);
	return solver;
}


void pogs_solve(pogs_solver * solver, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output)
{
	OK_TIMER t = tic();

	/* copy settings */
	update_settings(solver->settings, settings);

	/* copy and scale function vectors */
	update_problem(solver, f, g);

	/* get warm start variables */
	if (settings->warmstart)
		initialize_variables(solver);
	if ( !(settings->resume) )
		solver->rho = settings->rho;

	info->setup_time = toc(t);
	if (!(settings->warmstart || settings->resume))
		info->setup_time += solver->init_time;

	/* run solver */
	t = tic();
	pogs_solver_loop(solver, info);
	info->solve_time = toc(t);

	/* unscale output */
	copy_output(solver, output);
}

void pogs_finish(pogs_solver * solver, int reset)
{
	pogs_solver_free(solver);
	if (reset) ok_device_reset();
}

void pogs(ok_float * A, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	enum CBLAS_ORDER ord, EQUILIBRATOR equil, int reset)
{
	pogs_solver * solver = OK_NULL;
	solver = pogs_init(A, f->size, g->size, ord, equil);
	pogs_solve(solver, f, g, settings, info, output);
	pogs_finish(solver, reset);
}


pogs_solver * pogs_load_solver(ok_float * A_equil, ok_float * LLT_factorization,
	ok_float * d, ok_float * e, ok_float * z, ok_float * z12,
	ok_float * z_dual, ok_float * z_dual12, ok_float * z_prev, ok_float rho,
	size_t m, size_t n, enum CBLAS_ORDER ord)
{
	pogs_solver * solver = OK_NULL;
	pogs_solver_alloc(&solver, m , n, ord);

	matrix_memcpy_ma(solver->M->A, A_equil, ord);

	#ifndef OPTKIT_INDIRECT
	matrix_memcpy_ma(solver->M->P->L, LLT_factorization, ord);
	#endif

	vector_memcpy_va(solver->M->d, d, 1);
	vector_memcpy_va(solver->M->e, e, 1);

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
	matrix_memcpy_am(A_equil, solver->M->A, ord);

	#ifndef OPTKIT_INDIRECT
	matrix_memcpy_am(LLT_factorization, solver->M->P->L, ord);
	#endif

	vector_memcpy_av(d, solver->M->d, 1);
	vector_memcpy_av(e, solver->M->e, 1);

	vector_memcpy_av(z, solver->z->primal->vec, 1);
	vector_memcpy_av(z12, solver->z->primal12->vec, 1);
	vector_memcpy_av(z_dual, solver->z->dual->vec, 1);
	vector_memcpy_av(z_dual12, solver->z->dual12->vec, 1);
	vector_memcpy_av(z_prev, solver->z->prev->vec, 1);

	*rho = solver->rho;
}

#ifdef __cplusplus
}
#endif
