#include "optkit_pogs.h"

#ifdef __cplusplus
extern "C" {
#endif

void 
__pogs_matrix_alloc(pogs_matrix * M, size_t m, size_t n, CBLAS_ORDER_t ord){
	M = (pogs_matrix *) malloc( sizeof(pogs_matrix) );
	M->A = (matrix *) malloc( sizeof(matrix) );
	matrix_calloc(M->A, m, n, ord);
	M->P = (projector *) malloc( sizeof(projector) );
	PROJECTOR(alloc)(M->P, M->A);
	M->d = (vector *) malloc( sizeof(vector) );
	M->e = (vector *) malloc( sizeof(vector) );
	vector_calloc(M->d, m);
	vector_calloc(M->e, m);
	M->skinny = (m >= n);
	M->normalized = 0;
	M->equilibrated = 0;
	return;
}

void 
__pogs_matrix_free(pogs_matrix * M){
	PROJECTOR(free)(M->P);
	ok_free(M->P);
	matrix_free(M->A);
	vector_free(M->d);
	vector_free(M->e);
	ok_free(M);
}


void
__block_vector_alloc(block_vector * z, size_t m, size_t n){
	z = (block_vector *) malloc ( sizeof(block_vector) );
	z->size = m + n;
	z->m = m;
	z->n = n;
	z->vec = (vector *) malloc( sizeof(vector) );
	z->x = (vector *) malloc( sizeof(vector) );
	z->y = (vector *) malloc( sizeof(vector) );
	vector_calloc(z->vec, m + n);
	vector_subvector(z->y, z->vec, 0, m);
	vector_subvector(z->x, z->vec, m, n);
}

void
__block_vector_free(block_vector * z){
	vector_free(z->vec);
	ok_free(z->x);
	ok_free(z->y);
	ok_free(z->vec);
	ok_free(z);
}

void
__pogs_variables_alloc(pogs_variables * z, size_t m, size_t n){
	z = (pogs_variables *) malloc( sizeof(pogs_variables) );
	__block_vector_alloc(z->primal, m, n);
	__block_vector_alloc(z->primal12, m, n);
	__block_vector_alloc(z->dual, m ,n);
	__block_vector_alloc(z->dual12, m, n);
	__block_vector_alloc(z->prev, m ,n);
	__block_vector_alloc(z->temp, m, n);
}

void
__pogs_variables_free(pogs_variables * z){
	__block_vector_free(z->primal);
	__block_vector_free(z->primal12);
	__block_vector_free(z->dual);
	__block_vector_free(z->dual12);
	__block_vector_free(z->prev);
	__block_vector_free(z->temp);
	ok_free(z);
}



void
__pogs_solver_alloc(pogs_solver * solver, size_t m, size_t n, 
	CBLAS_ORDER_t ord){

	solver = (pogs_solver *) malloc( sizeof(pogs_solver) );
	solver->settings = (pogs_settings *) malloc( sizeof(pogs_settings) );
	solver->f = (FunctionVector *) malloc( sizeof(FunctionVector) );
	solver->g = (FunctionVector *) malloc( sizeof(FunctionVector) );	
	function_vector_calloc(solver->f, m);
	function_vector_calloc(solver->g, n);
	__pogs_variables_alloc(solver->z, m, n);
	__pogs_matrix_alloc(solver->M, m, n, ord);
	blas_make_handle(&(solver->linalg_handle));
	solver->rho = kOne;
}

void
__pogs_solver_free(pogs_solver * solver){
	blas_destroy_handle(solver->linalg_handle);
	__pogs_matrix_free(solver->M);
	__pogs_variables_free(solver->z);
	ok_free(solver->settings);
	function_vector_free(solver->f);
	function_vector_free(solver->g);
	ok_free(solver->f);
	ok_free(solver->g);	
	ok_free(solver);
}


void 
__update_settings(pogs_settings * settings, const pogs_settings * input){
	memcpy((void *) settings, (void *) input, sizeof(pogs_settings));
}


void
__equilibrate(void * linalg_handle, ok_float * A_orig, pogs_matrix * M, 
	Equilibration_t equil, CBLAS_ORDER_t ord){

	if (equil == EquilSinkhorn)
		sinkhorn_knopp(linalg_handle, A_orig, M->A, M->d, M->e, ord);
	else
		dense_l2(linalg_handle, A_orig, M->A, M->d, M->e, ord);
	M->equilibrated = 1;
}

ok_float
__estimate_norm(void * linalg_handle, pogs_matrix * M){
	return kOne;
}

void
__normalize_DAE(void * linalg_handle, pogs_matrix * M){
	size_t m = M->A->size1,  n = M->A->size2; 
	size_t mindim = m < n ? m : n;
	ok_float factor;

	M->normalized = M->P->normalized;
	M->normA = M->P->normA;


	if (!(M->normalized)){
		M->normA = __estimate_norm(linalg_handle, M) / MATH(sqrt)(mindim);
		matrix_scale(M->A, kOne / M->normA);
		M->normalized = 1;
	}
	factor = MATH(sqrt)(MATH(sqrt)( 
		(n * blas_dot(linalg_handle, M->d, M->d)) / 
		(m * blas_dot(linalg_handle, M->e, M->e))  ));
	vector_scale(M->d, MATH(sqrt)(M->normA) / factor);
	vector_scale(M->e, factor / MATH(sqrt)(M->normA));
}


void 
__scale_problem(FunctionVector * f, FunctionVector * g, 
	vector * d, vector * e){

	int i;
	size_t m = f->size, n=g->size;
	vector params = (vector){0, 0, OK_NULL};
	params.stride = sizeof(FunctionObj) / sizeof(ok_float);

	/* rescale f parameters f->a, f->c, f->e */
	params.size = m;
	for (i = 0; i < 3; ++i){
		params.data = function_vector_get_parameteraddress(f, i);
		vector_div(&params, d);
	}

	/* rescale g parameters g->a, g->d, g->e */
	params.size = n;
	for (i = 0; i < 3; ++i){
		params.data = function_vector_get_parameteraddress(g, i);
		vector_mul(&params, e);
	}
}

void
__initialize_variables(pogs_solver * solver){
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
		vector_mul(z->dual->y, solver->M->d);
		blas_gemv(solver->linalg_handle, CblasTrans, -kOne,
			solver->M->A, z->dual->y, kZero, z->dual->x);
		vector_scale(z->dual->vec, -kOne / solver->rho);
	}
}



pogs_tolerances  
__make_tolerances(const pogs_settings * settings, size_t m, size_t n){
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


void
__update_objective(void * linalg_handle, 
	FunctionVector * f, FunctionVector * g, 
	ok_float rho, pogs_variables * z, pogs_objectives * obj){

	obj->gap = rho * MATH(fabs)( 
		blas_dot(linalg_handle, z->primal12->vec, z->dual12->vec) );
	obj->primal = FuncEvalVector(f, z->primal12->y) + 
			FuncEvalVector(g, z->primal12->x);
	obj->dual = obj->primal - obj->gap;
}

void 
__update_tolerances(void * linalg_handle, pogs_variables * z, 
	pogs_objectives * obj, pogs_tolerances * eps){

	/* ------------------------------------------------------------	*/
	/* eps_primal 	= eps_abs + eps_rel * sqrt(m) * || y^k || 		*/
	/* eps_dual 	= eps_abs + eps_rel * sqrt(n) * || x^k || 		*/
	/* eps_gap	 	= eps_abs + eps_rel * sqrt(mn) * (f(y) + g(x)) 	*/
	/* ------------------------------------------------------------ */
	eps->primal = eps->atolm + eps->reltol * MATH(sqrt)(blas_dot(linalg_handle, 
		z->primal->y, z->primal->y));
	eps->dual = eps->atoln + eps->reltol * MATH(sqrt)(blas_dot(linalg_handle,
		z->dual->x, z->dual->x));
	eps->gap = eps->atolmn + eps->reltol * MATH(fabs)(obj->primal);

}

int
__update_residuals(void * linalg_handle, pogs_solver * solver, 
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps, 
	int force_exact){

	pogs_variables * z = solver->z;
	matrix * A = solver->M->A;

	/* approximate residuals */
	if ( !force_exact ){
		/* -------------------------------- */
		/* res_dual ~ || z^k - z^(k-1) ||_2 */
		/* -------------------------------- */
		vector_memcpy_vv(z->temp->vec, z->prev->vec);
		blas_axpy(linalg_handle, -kOne, z->primal->vec, z->temp->vec);
		res->dual = MATH(sqrt)(blas_dot(linalg_handle, 
			z->temp->vec, z->temp->vec));

		/* -------------------------------------- */
		/* res_primal ~ || z^(k + 1/2) - z^k ||_2 */
		/* -------------------------------------- */
		vector_memcpy_vv(z->temp->vec, z->primal12->vec);
		blas_axpy(linalg_handle, -kOne, z->primal->vec, z->temp->vec);
		res->primal = MATH(sqrt)(blas_dot(linalg_handle, 
			z->temp->vec, z->temp->vec));
	}

	/* exact residuals */
	res->gap = obj->gap;


	if ( force_exact || 
		((res->primal < eps->primal) && (res->dual < eps->dual)) ) {
		/* -------------------------------------------------------- */
		/* ||Ax^(k+1/2) - y^(k+1/2)|| <=? eps_rel * || y^(k+1/2)||  */
		/* -------------------------------------------------------- */

		vector_memcpy_vv(z->temp->y, z->primal12->y);
		blas_gemv(linalg_handle, CblasNoTrans, kOne, 
			A, z->primal12->x, -kOne, z->temp->y);
		res->primal = MATH(sqrt)(blas_dot(linalg_handle, 
			z->temp->y, z->temp->y));

		/* ------------------------------------------------------------ */
		/* ||A'yt^(k+1/2) - xt^(k+1/2)|| <=? eps_rel * || xt^(k+1/2)||  */
		/* ------------------------------------------------------------ */
		if (res->primal < eps->primal){
			vector_memcpy_vv(z->temp->x, z->dual12->x);
			blas_gemv(linalg_handle, CblasTrans, kOne, 
				A, z->dual12->y, kOne, z->temp->x);
			res->dual = solver->rho * MATH(sqrt)(blas_dot(linalg_handle, 
				z->temp->x, z->temp->x));
			return 1;
		}
	}
	return 0;

}

int 
__check_convergence(void * linalg_handle, pogs_solver * solver, 
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps, 
	int gapstop, int force_exact){

	int exact;

	__update_objective(linalg_handle, solver->f, solver->g, 
		solver->rho, solver->z, obj);
	__update_tolerances(linalg_handle, solver->z, obj, eps);
	exact = __update_residuals(linalg_handle, solver, obj, res, eps,
		force_exact);

	exact &= (res->primal < eps->primal);
	exact &= (res->dual < eps->dual);
	exact &= (res->gap < eps->gap || !(gapstop));

	return exact;
}


/* --------------------------------------- */
/* z_out <- alpha * z_12 + (1 - alpha) * z */
/* --------------------------------------- */
void 
__overrelax(void * linalg_handle, vector * z12, vector * z, vector * z_out, 
	ok_float alpha, int overwrite){

	if (alpha != kOne){
		if (overwrite){
			vector_memcpy_vv(z_out, z12);
			vector_scale(z_out, alpha);
		} else {
			blas_axpy(linalg_handle, alpha, z12, z_out);
		}
		blas_axpy(linalg_handle, kOne - alpha, z, z_out);

	} else 
		vector_memcpy_vv(z_out, z12);
}


/* ------------------------------- */
/* y^{k+1/2} = Prox_{rho, f} (y^k) */
/* x^{k+1/2} = Prox_{rho, g} (x^k) */
/* ------------------------------- */

void
__prox(void * linalg_handle, FunctionVector * f, 
	FunctionVector * g, pogs_variables * z, ok_float rho){

	vector_memcpy_vv(z->temp->vec, z->primal->vec);
	blas_axpy(linalg_handle, -kOne, z->dual->vec, z->temp->vec);
	ProxEvalVector(f, rho, z->temp->y, z->primal12->y);
	ProxEvalVector(g, rho, z->temp->x, z->primal12->x);
}

/* ---------------------------------------------------------- */
/* ( x^{k+1}, y^{k+1} ) = Proj_{y=Ax} ( x^{k+1/2, y^{k+1/2} ) */
/* ---------------------------------------------------------- */
void 
__project_primal(void * linalg_handle, projector * proj, 
	pogs_variables * z,  ok_float alpha){

	__overrelax(linalg_handle, z->primal12->vec, z->prev->vec, 
		z->temp->vec, alpha, 1);
	PROJECTOR(project)(linalg_handle, proj, z->primal12->x, z->primal12->y,
		z->primal->x, z->primal->y);
}


/* ------------------------------ */
/* update zt^{k+1/2} and zt^{k+1} */
/* ------------------------------ */
void
__update_dual(void * linalg_handle, pogs_variables * z, ok_float alpha){
	
	/* ------------------------------------ */
	/* zt^{k+1/2} = z^{k+1/2} - z^k + zt^k 	*/
	/* ------------------------------------ */
	vector_memcpy_vv(z->dual12->vec, z->primal12->vec);
	blas_axpy(linalg_handle, -kOne, z->prev->vec, z->dual12->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->dual->vec);

	/* ------------------------------------ */
	/* zt^{k+1}   = z^{k+1/2} - z^k 		*/
	/* ------------------------------------ */
	__overrelax(linalg_handle, z->primal12->vec, z->prev->vec, 
		z->dual->vec, alpha, 0);
	blas_axpy(linalg_handle, -kOne, z->primal->vec, z->dual->vec);

}


/* --------------------------------------------------------- */
/* change solver->rho to balance primal and dual convergence */
/* (and rescale z->dual accordingly) 						 */
/* --------------------------------------------------------- */
void
__adaptrho(pogs_solver * solver, adapt_params * params, 
	pogs_residuals * res, pogs_tolerances * eps, uint k){

	if (!(solver->settings->adaptiverho)) return;
	pogs_settings * settings = solver->settings;


	if (res->dual < params->xi * eps->dual && 
		res->primal < params->xi*eps->primal){
		params->xi *= kKAPPA;
	} else if (res->dual < params->xi * eps->dual &&  
		res->primal > params->xi * eps->primal && kTAU * k > params->l){
		if (solver->rho < kRHOMAX){
			solver->rho *= params->delta;
			if (settings->verbose > 2)
			printf("+RHO %0.3ed\n", (float) solver->rho);
			vector_scale(solver->z->dual->vec, kOne / params->delta);
			params->delta = params->delta * kGAMMA < kDELTAMAX?
				params->delta * kGAMMA : kDELTAMAX;
			params->u = k;
		}
	} else if (res->primal < params->xi * eps->primal && 
		res->dual > params->xi * eps->dual && kTAU * k > params->u){
		if (solver->rho > kRHOMIN){
			solver->rho /= params->delta;
			if (settings->verbose > 2)
				printf("-RHO%0.3e\n", (float) solver->rho);
			vector_scale(solver->z->dual->vec, params->delta);
			params->delta = params->delta * kGAMMA < kDELTAMAX ?
				params->delta * kGAMMA : kDELTAMAX;
			params->l = k;
		}
	} else {
		params->delta = params->delta / kGAMMA > kDELTAMIN ? 
			params->delta / kGAMMA : kDELTAMIN;
	}
}



void 
__copy_output(pogs_solver * solver, pogs_output * output){
	vector * d = solver->M->d;
	vector * e = solver->M->e;
	pogs_variables * z = solver->z;

	/* ------------- */
	/* x = x^(k+1/2) */
	/* y = y^(k+1/2) */
	/* ------------- */
	vector_memcpy_vv(z->temp->vec, z->primal12->vec);
	vector_mul(z->temp->x, e);
	vector_div(z->temp->y, d);
	vector_memcpy_av(output->x, z->temp->x, 1);
	vector_memcpy_av(output->y, z->temp->y, 1);

	/* ---------------------- */
	/* mu = -rho * xt^(k+1/2) */
	/* nu = -rho * yt^(k+1/2) */
	/* ---------------------- */
	vector_memcpy_vv(z->temp->vec, z->dual12->vec);
	vector_scale(z->temp->vec, -solver->rho);
	vector_div(z->temp->x, e);
	vector_mul(z->temp->y, d);
	vector_memcpy_av(output->mu, z->temp->x, 1);
	vector_memcpy_av(output->nu, z->temp->y, 1);
}


void 
__print_header_string(){
	printf("   #  res_pri     eps_pri   res_dual   eps_dual\
	   gap        eps_gap    objective\n");
}

void
__print_iter_string(pogs_residuals * res, pogs_tolerances * eps, 
	pogs_objectives * obj, uint k){
	printf("   %u: %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e\n",
		k, res->primal, eps->primal, res->dual, eps->dual, 
		res->gap, eps->gap, obj->primal);
}

void
__pogs_solver_loop(pogs_solver * solver, pogs_info * info){

	/* declare / get handles to all auxiliary types */
	int err = 0, converged = 0;
	uint k, PRINT_ITER = 10000u / (10u^solver->settings->verbose);
	adapt_params rho_params = (adapt_params){kDELTAMIN, kZero, kZero, kOne};
	pogs_settings * settings = solver->settings;
	pogs_variables * z = solver->z;
	pogs_objectives obj = (pogs_objectives){NAN, NAN, NAN};
	pogs_residuals res = (pogs_residuals){NAN, NAN, NAN};
	pogs_tolerances eps = __make_tolerances(settings, 
		solver->z->m, solver->z->n);
	void * linalg_handle = solver->linalg_handle;

	if (settings->verbose == 0)
		PRINT_ITER = settings->maxiter * 2u;

	/* signal start of execution */
	if (!err && settings->verbose > 0)
		__print_header_string();

	/* iterate until converged, or error/maxiter reached */
	for (k = 1; !(converged || err); ++k){
		__prox(linalg_handle, solver->f, solver->g, z, solver->rho);
		__project_primal(linalg_handle, solver->M->P, z, settings->alpha);
		__update_dual(linalg_handle, z, settings->alpha);
	
		converged = __check_convergence(linalg_handle, solver,
			&obj, &res, &eps, settings->gapstop, 
			(settings->resume && k == 1));

		if (settings->verbose > 0 && (k % PRINT_ITER == 0 || converged))
			__print_iter_string(&res, &eps, &obj, k);

		if (settings->adaptiverho)
			__adaptrho(solver, &rho_params, &res, &eps, k);
	}	

	if (!converged && k==settings->maxiter)
		printf("reached max iter = %u\n", k);

	/* update info */
	info->rho = solver->rho;
	info->obj = obj.primal;
	info->converged = converged;
	info->err = err;
	info->k = k;
}



void
set_default_settings(pogs_settings * s){
	__update_settings(s, &kDefaultPOGSSettings);
}

void
pogs_init(pogs_solver * solver, ok_float * A, size_t m, size_t n, 
	CBLAS_ORDER_t ord, Equilibration_t equil){

	/* make variables, matrix */
	__pogs_solver_alloc(solver, m , n, ord);

	/* equilibrate A as (D * A_equil * E) = A */
	__equilibrate(solver->linalg_handle, A, solver->M, equil, ord);

	/* make projector; normalize A; adjust d, e accordingly */
	PROJECTOR(initialize)(solver->linalg_handle, solver->M->P, 1);
	__normalize_DAE(solver->linalg_handle, solver->M);	
}


void 
pogs_solve(pogs_solver * solver, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output){

	__update_settings(solver->settings, settings);

	/* copy function vectors */
	function_vector_memcpy_va(solver->f, f->objectives);
	function_vector_memcpy_va(solver->g, g->objectives);

	/* scale objectives */
	__scale_problem(solver->f, solver->g, solver->M->d, solver->M->e);

	/* get warm start variables */
	if (settings->warmstart)
		__initialize_variables(solver);
	if ( !(settings->resume) )
		solver->rho = settings->rho;

	/* run solver */
	__pogs_solver_loop(solver, info);

	/* unscale output */
	__copy_output(solver, output);
}

void 
pogs_finish(pogs_solver * solver){
	__pogs_solver_free(solver);
}

void
pogs(ok_float * A, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	CBLAS_ORDER_t ord, Equilibration_t equil){
	pogs_solver * solver = OK_NULL;
	pogs_init(solver, A, f->size, g->size, ord, equil);
	pogs_solve(solver, f, g, settings, info, output);
	pogs_finish(solver);
}


void 
pogs_load_solver(pogs_solver * solver, ok_float * A_equil, 
	ok_float * LLT_factorization, ok_float * d, 
	ok_float * e, ok_float * z, ok_float * z12, ok_float * z_dual, 
	ok_float * z_dual12, ok_float * z_prev, ok_float rho){


	matrix_memcpy_ma(solver->M->A, A_equil, solver->M->A->rowmajor);

	#ifndef OPTKIT_INDIRECT
	matrix_memcpy_ma(solver->M->P->L, LLT_factorization, 
		solver->M->A->rowmajor);
	#endif

	vector_memcpy_va(solver->M->d, d, 1);
	vector_memcpy_va(solver->M->e, e, 1);

	vector_memcpy_va(solver->z->primal->vec, z, 1);
	vector_memcpy_va(solver->z->primal12->vec, z12, 1);
	vector_memcpy_va(solver->z->dual->vec, z_dual, 1);
	vector_memcpy_va(solver->z->dual12->vec, z_dual12, 1);
	vector_memcpy_va(solver->z->prev->vec, z_prev, 1);

	solver->rho = rho;
}

void 
pogs_extract_solver(pogs_solver * solver, ok_float * A_equil, 
	ok_float * LLT_factorization, ok_float * d, 
	ok_float * e, ok_float * z, ok_float * z12, ok_float * z_dual, 
	ok_float * z_dual12, ok_float * z_prev, ok_float * rho){

	matrix_memcpy_am(A_equil, solver->M->A, solver->M->A->rowmajor);

	#ifndef OPTKIT_INDIRECT
	matrix_memcpy_am(LLT_factorization, solver->M->P->L, 
		solver->M->A->rowmajor);
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