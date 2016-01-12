#include "optkit_pogs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------------- */
/* POGS private API:																*/
/*																					*/
/*	void pogs_matrix_alloc(pogs_matrix ** M, size_t m, size_t n, CBLAS_ORDER_t ord) */
/*	void pogs_matrix_free(pogs_matrix * M)											*/
/*	void block_vector_alloc(block_vector ** z, size_t m, size_t n) 					*/
/* 	void block_vector_free(block_vector * z)										*/
/* 	void pogs_variables_alloc(pogs_variables ** z, size_t m, size_t n) 				*/
/*	void pogs_variables_free(pogs_variables * z) 									*/
/*  void pogs_solver_alloc(pogs_solver ** solver, size_t m, size_t n, 				*/ 
/*		CBLAS_ORDER_t ord)															*/
/*	void pogs_solver_free(pogs_solver * solver)										*/
/*	void update_settings(pogs_settings * settings, const pogs_settings * input) 	*/
/* 	void updatae_problem(pogs_solver * solver, 										*/ 
/*		FunctionVector * f, FunctionVector *g) 										*/
/*	void equilibrate(void * linalg_handle, ok_float * A_orig, pogs_matrix * M, 		*/
/*		Equilibration_t equil, CBLAS_ORDER_t ord) 									*/
/*		ok_float estimate_norm(void * linalg_handle, pogs_matrix * M) 				*/
/*	void normalize_DAE(void * linalg_handle, pogs_matrix * M) 						*/
/*	void initialize_variables(pogs_solver * solver) 								*/
/*	pogs_tolerances make_tolerances(const pogs_settings * settings, 				*/
/*		size_t m, size_t n) 														*/
/*	void update_objective(void * linalg_handle,  									*/
/*		FunctionVector * f, FunctionVector * g, 									*/
/*		ok_float rho, pogs_variables * z, pogs_objectives * obj) 					*/
/*	void update_tolerances(void * linalg_handle, pogs_variables * z, 				*/
/*		pogs_objectives * obj, pogs_tolerances * eps)								*/
/*	void update_residuals(void * linalg_handle, pogs_solver * solver, 				*/
/*		pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps, 		*/
/*		int force_exact) 															*/
/*	int check_convergence(void * linalg_handle, pogs_solver * solver, 				*/
/*		pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps, 		*/
/*		int gapstop, int force_exact) 												*/
/*	void set_prev(pogs_variables * z) 												*/
/*	void prox(void * linalg_handle, FunctionVector * f, 							*/
/*		FunctionVector * g, pogs_variables * z, ok_float rho) 						*/
/*	void project_primal(void * linalg_handle, projector * proj, 					*/
/*		pogs_variables * z,  ok_float alpha)										*/
/*	void update_dual(void * linalg_handle, pogs_variables * z, ok_float alpha) 		*/
/*	void adaptrho(pogs_solver * solver, adapt_params * params, 						*/
/*		pogs_residuals * res, pogs_tolerances * eps, uint k) 						*/
/*	void copy_output(pogs_solver * solver, pogs_output * output) 					*/
/*	void print_header_string() 														*/
/*	void print_iter_string(pogs_residuals * res, pogs_tolerances * eps, 			*/
/*		pogs_objectives * obj, uint k) 												*/
/*	void pogs_solver_loop(pogs_solver * solver, pogs_info * info) 					*/
/* -------------------------------------------------------------------------------- */

int
private_api_accessible(){
#ifdef OK_DEBUG_PYTHON
return 1;
#else
return 0;
#endif
}

int
is_direct(){
	return sizeof(projector) == sizeof(direct_projector);
}

void 
POGS(pogs_matrix_alloc)(pogs_matrix ** M, size_t m, size_t n, CBLAS_ORDER_t ord){
	pogs_matrix * M_;
	M_ = (pogs_matrix *) malloc( sizeof(pogs_matrix) );
	M_->A = (matrix *) malloc( sizeof(matrix) );
	matrix_calloc(M_->A, m, n, ord);
	M_->P = (projector *) malloc( sizeof(projector) );
	PROJECTOR(alloc)(M_->P, M_->A);
	M_->d = (vector *) malloc( sizeof(vector) );
	M_->e = (vector *) malloc( sizeof(vector) );
	vector_calloc(M_->d, m);
	vector_calloc(M_->e, n);
	M_->skinny = (m >= n);
	M_->normalized = 0;
	M_->equilibrated = 0;
	*M = M_;
}

void 
POGS(pogs_matrix_free)(pogs_matrix * M){
	PROJECTOR(free)(M->P);
	ok_free(M->P);
	matrix_free(M->A);
	vector_free(M->d);
	vector_free(M->e);
	ok_free(M);
}


void
POGS(block_vector_alloc)(block_vector ** z, size_t m, size_t n){
	block_vector * z_ = OK_NULL;
	z_ = (block_vector *) malloc ( sizeof(block_vector) );
	z_->size = m + n;
	z_->m = m;
	z_->n = n;
	z_->vec = (vector *) malloc( sizeof(vector) );
	z_->x = (vector *) malloc( sizeof(vector) );
	z_->y = (vector *) malloc( sizeof(vector) );
	vector_calloc(z_->vec, m + n);
	vector_subvector(z_->y, z_->vec, 0, m);
	vector_subvector(z_->x, z_->vec, m, n);
	* z = z_;
}

void
POGS(block_vector_free)(block_vector * z){
	vector_free(z->vec);
	ok_free(z->x);
	ok_free(z->y);
	ok_free(z->vec);
	ok_free(z);
}

void
POGS(pogs_variables_alloc)(pogs_variables ** z, size_t m, size_t n){
	pogs_variables * z_ = OK_NULL;
	z_ = (pogs_variables *) malloc( sizeof(pogs_variables) );
	z_->m = m;
	z_->n = n;
	POGS(block_vector_alloc)(&(z_->primal), m, n);
	POGS(block_vector_alloc)(&(z_->primal12), m, n);
	POGS(block_vector_alloc)(&(z_->dual), m ,n);
	POGS(block_vector_alloc)(&(z_->dual12), m, n);
	POGS(block_vector_alloc)(&(z_->prev), m ,n);
	POGS(block_vector_alloc)(&(z_->temp), m, n);
	*z = z_;
}

void
POGS(pogs_variables_free)(pogs_variables * z){
	POGS(block_vector_free)(z->primal);
	POGS(block_vector_free)(z->primal12);
	POGS(block_vector_free)(z->dual);
	POGS(block_vector_free)(z->dual12);
	POGS(block_vector_free)(z->prev);
	POGS(block_vector_free)(z->temp);
	ok_free(z);
}

void
POGS(pogs_solver_alloc)(pogs_solver ** solver, size_t m, size_t n, 
	CBLAS_ORDER_t ord){

	pogs_solver * s = OK_NULL;

	s = (pogs_solver *) malloc( sizeof(pogs_solver) );
	s->settings = (pogs_settings *) malloc( sizeof(pogs_settings) );
	s->f = (FunctionVector *) malloc( sizeof(FunctionVector) );
	s->g = (FunctionVector *) malloc( sizeof(FunctionVector) );	
	s->f->objectives = OK_NULL;
	s->g->objectives = OK_NULL;
	function_vector_calloc(s->f, m);
	function_vector_calloc(s->g, n);
	POGS(pogs_variables_alloc)(&(s->z), m, n);
	POGS(pogs_matrix_alloc)(&(s->M), m, n, ord);
	blas_make_handle(&(s->linalg_handle));
	s->rho = kOne;
	* solver = s;
}

void
POGS(pogs_solver_free)(pogs_solver * solver){
	blas_destroy_handle(solver->linalg_handle);
	POGS(pogs_matrix_free)(solver->M);
	POGS(pogs_variables_free)(solver->z);
	ok_free(solver->settings);
	function_vector_free(solver->f);
	function_vector_free(solver->g);
	ok_free(solver->f);
	ok_free(solver->g);	
	ok_free(solver);
}


void 
POGS(update_settings)(pogs_settings * settings, const pogs_settings * input){
	memcpy((void *) settings, (void *) input, sizeof(pogs_settings));
}


void
POGS(equilibrate)(void * linalg_handle, ok_float * A_orig, pogs_matrix * M, 
	Equilibration_t equil, CBLAS_ORDER_t ord){

	if (equil == EquilSinkhorn)
		sinkhorn_knopp(linalg_handle, A_orig, M->A, M->d, M->e, ord);
	else
		dense_l2(linalg_handle, A_orig, M->A, M->d, M->e, ord);
	M->equilibrated = 1;
}

ok_float
POGS(estimate_norm)(void * linalg_handle, pogs_matrix * M){
	return kOne;
}

void
POGS(normalize_DAE)(void * linalg_handle, pogs_matrix * M){
	size_t m = M->A->size1,  n = M->A->size2; 
	size_t mindim = m < n ? m : n;
	ok_float factor;

	M->normalized = M->P->normalized;
	M->normA = M->P->normA;


	if (!(M->normalized)){
		M->normA = POGS(estimate_norm)(linalg_handle, M) /
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


void 
POGS(update_problem)(pogs_solver * solver, FunctionVector * f, 
	FunctionVector * g){

	function_vector_memcpy_va(solver->f, f->objectives);
	function_vector_memcpy_va(solver->g, g->objectives);
	function_vector_div(solver->f, solver->M->d);
	function_vector_mul(solver->g, solver->M->e);
}

void
POGS(initialize_variables)(pogs_solver * solver){
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



pogs_tolerances  
POGS(make_tolerances)(const pogs_settings * settings, size_t m, size_t n){
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
POGS(update_objective)(void * linalg_handle, 
	FunctionVector * f, FunctionVector * g, 
	ok_float rho, pogs_variables * z, pogs_objectives * obj){

	obj->gap = MATH(fabs)( 
		blas_dot(linalg_handle, z->primal12->vec, z->dual12->vec) );
	obj->primal = FuncEvalVector(f, z->primal12->y) + 
			FuncEvalVector(g, z->primal12->x);
	obj->dual = obj->primal - obj->gap;
}

void 
POGS(update_tolerances)(void * linalg_handle, pogs_variables * z, 
	pogs_objectives * obj, pogs_tolerances * eps){

	/* ------------------------------------------------------------	*/
	/* eps_primal 	= eps_abs + eps_rel * sqrt(m) * ||y^k|| 		*/
	/* eps_dual 	= eps_abs + eps_rel * sqrt(n) * ||x^k|| 		*/
	/* eps_gap	 	= eps_abs + eps_rel * sqrt(mn) *  				*/
	/*										||z^k|| * ||z^k+1/2||	*/
	/* ------------------------------------------------------------ */
	eps->primal = eps->atolm + eps->reltol * MATH(sqrt)(blas_dot(linalg_handle, 
		z->primal12->y, z->primal12->y));
	eps->dual = eps->atoln + eps->reltol * MATH(sqrt)(blas_dot(linalg_handle,
		z->dual12->x, z->dual12->x));
	eps->gap = eps->atolmn + eps->reltol * MATH(sqrt)(blas_dot(linalg_handle,
		z->primal->vec, z->primal->vec)) * MATH(sqrt)(blas_dot(linalg_handle,
		z->primal12->vec, z->primal12->vec));

}

void
POGS(update_residuals)(void * linalg_handle, pogs_solver * solver, 
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps){

	pogs_variables * z = solver->z;
	matrix * A = solver->M->A;

	res->gap = obj->gap;

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
	vector_memcpy_vv(z->temp->x, z->dual12->x);
	blas_gemv(linalg_handle, CblasTrans, kOne, 
		A, z->dual12->y, kOne, z->temp->x);
	res->dual = MATH(sqrt)(blas_dot(linalg_handle, 
		z->temp->x, z->temp->x));
}

int 
POGS(check_convergence)(void * linalg_handle, pogs_solver * solver, 
	pogs_objectives * obj, pogs_residuals * res, pogs_tolerances * eps, 
	int gapstop){



	POGS(update_objective)(linalg_handle, solver->f, solver->g, 
		solver->rho, solver->z, obj);
	POGS(update_tolerances)(linalg_handle, solver->z, obj, eps);
	POGS(update_residuals)(linalg_handle, solver, obj, res, eps);

	return (res->primal < eps->primal) && (res->dual < eps->dual) &&
		(res->gap < eps->gap || !(gapstop));
}


/* -------------- */
/* z^k <- z^{k+1} */
/* -------------- */
void
POGS(set_prev)(pogs_variables * z){
	vector_memcpy_vv(z->prev->vec, z->primal->vec);
}

/* -------------------------------------- */
/* y^{k+1/2} = Prox_{rho, f} (y^k - yt^k) */
/* x^{k+1/2} = Prox_{rho, g} (x^k - xt^k) */
/* -------------------------------------- */

void
POGS(prox)(void * linalg_handle, FunctionVector * f, 
	FunctionVector * g, pogs_variables * z, ok_float rho){

	vector_memcpy_vv(z->temp->vec, z->primal->vec);
	blas_axpy(linalg_handle, -kOne, z->dual->vec, z->temp->vec);
	ProxEvalVector(f, rho, z->temp->y, z->primal12->y);
	ProxEvalVector(g, rho, z->temp->x, z->primal12->x);
}

/* --------------------------------------------------------	*/
/* ( x^{k+1}, y^{k+1} ) = Proj_{y=Ax} ( x^{k+1/2 + xt^k,	*/	
/*									y^{k+1/2} + yt^k) 		*/
/* -------------------------------------------------------- */
void 
POGS(project_primal)(void * linalg_handle, projector * proj, 
	pogs_variables * z,  ok_float alpha){

	vector_set_all(z->temp->vec, kZero);
	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->temp->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->temp->vec);
	PROJECTOR(project)(linalg_handle, proj, z->temp->x, z->temp->y,
		z->primal->x, z->primal->y);
}


/* ------------------------------ */
/* update zt^{k+1/2} and zt^{k+1} */
/* ------------------------------ */
void
POGS(update_dual)(void * linalg_handle, pogs_variables * z, ok_float alpha){
	
	/* ------------------------------------ */
	/* zt^{k+1/2} = z^{k+1/2} - z^k + zt^k 	*/
	/* ------------------------------------ */
	vector_memcpy_vv(z->dual12->vec, z->primal12->vec);
	blas_axpy(linalg_handle, -kOne, z->prev->vec, z->dual12->vec);
	blas_axpy(linalg_handle, kOne, z->dual->vec, z->dual12->vec);


	/* -------------------------------------- */
	/* zt^{k+1}   = zt^k + z^{k+1/2} - z^k+1  */
	/* -------------------------------------- */
	blas_axpy(linalg_handle, alpha, z->primal12->vec, z->dual->vec);
	blas_axpy(linalg_handle, kOne - alpha, z->prev->vec, z->dual->vec);		
	blas_axpy(linalg_handle, -kOne, z->primal->vec, z->dual->vec);
}


/* --------------------------------------------------------- */
/* change solver->rho to balance primal and dual convergence */
/* (and rescale z->dual accordingly) 						 */
/* --------------------------------------------------------- */
void
POGS(adaptrho)(pogs_solver * solver, adapt_params * params, 
	pogs_residuals * res, pogs_tolerances * eps, uint k){

	if (!(solver->settings->adaptiverho)) return;

	if (res->dual < params->xi * eps->dual && 
		res->primal > params->xi * eps->primal &&
		kTAU * (ok_float) k > params->l) {
		
		if (solver->rho < kRHOMAX) {
		  solver->rho *= params->delta;
		  if (solver->settings->verbose > 2)
			  printf("+RHO: %.3e\n", solver->rho);
		  vector_scale(solver->z->dual->vec, kOne / params->delta);
		  params->delta = params->delta * kGAMMA < kDELTAMAX ?
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
		  params->delta = params->delta * kGAMMA < kDELTAMAX ?
		  					params->delta * kGAMMA : kDELTAMAX;
		  params->l = (ok_float) k; 
		}

	} else if (res->dual < params->xi * eps->dual && 
				res->primal < params->xi * eps->primal) {
		params->xi *= kKAPPA;
	} else {
		params->delta = params->delta / kGAMMA > kDELTAMIN ? 
						params->delta / kGAMMA : kDELTAMIN;
	}
}



void 
POGS(copy_output)(pogs_solver * solver, pogs_output * output){
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

	/* ----------------- */
	/* mu1 = -rho * xt^k */
	/* nu1 = -rho * yt^k */
	/* ----------------- */
	vector_memcpy_vv(z->temp->vec, z->dual->vec);
	vector_scale(z->temp->vec, -solver->rho);
	vector_div(z->temp->x, e);
	vector_mul(z->temp->y, d);
	vector_memcpy_av(output->mu1, z->temp->x, 1);
	vector_memcpy_av(output->nu1, z->temp->y, 1);	
}


void 
POGS(print_header_string)(){
	printf("   #\tres_pri    eps_pri    res_dual   eps_dual\
   gap        eps_gap    objective\n");
}

void
POGS(print_iter_string)(pogs_residuals * res, pogs_tolerances * eps, 
	pogs_objectives * obj, uint k){
	printf("   %u: %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e\n",
		k, res->primal, eps->primal, res->dual, eps->dual, 
		res->gap, eps->gap, obj->primal);
}

void
POGS(pogs_solver_loop)(pogs_solver * solver, pogs_info * info){

	/* declare / get handles to all auxiliary types */
	int err = 0, converged = 0;
	uint k, PRINT_ITER = 10000u;
	adapt_params rho_params = (adapt_params){kDELTAMIN, kZero, kZero, kOne};
	pogs_settings * settings = solver->settings;
	pogs_variables * z = solver->z;
	pogs_objectives obj = (pogs_objectives){NAN, NAN, NAN};
	pogs_residuals res = (pogs_residuals){NAN, NAN, NAN};
	pogs_tolerances eps = POGS(make_tolerances)(settings, 
		solver->z->m, solver->z->n);
	void * linalg_handle = solver->linalg_handle;

	if (settings->verbose == 0)
		PRINT_ITER = settings->maxiter * 2u;
	else
		for (k = 0; k < settings->verbose && PRINT_ITER > 1; ++k)
			PRINT_ITER /= 10;

	/* signal start of execution */
	if (!err && settings->verbose > 0)
		POGS(print_header_string)();

	/* iterate until converged, or error/maxiter reached */
	for (k = 1; !err; ++k){
		POGS(set_prev)(z);
		POGS(prox)(linalg_handle, solver->f, solver->g, z, solver->rho);
		POGS(project_primal)(linalg_handle, solver->M->P, z, settings->alpha);
		POGS(update_dual)(linalg_handle, z, settings->alpha);

		converged = POGS(check_convergence)(linalg_handle, solver,
			&obj, &res, &eps, settings->gapstop);

		if (settings->verbose > 0 && 
			(k % PRINT_ITER == 0 || converged || k == settings->maxiter))
			POGS(print_iter_string)(&res, &eps, &obj, k);

		if (converged || k == settings->maxiter) break;

		if (settings->adaptiverho)
			POGS(adaptrho)(solver, &rho_params, &res, &eps, k);
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



void
set_default_settings(pogs_settings * s){
	s->alpha = kALPHA; 
	s->rho = kOne;
	s->abstol = kATOL;
	s->reltol = kRTOL;
	s->maxiter = kMAXITER;
	s->verbose = kVERBOSE;
	s->adaptiverho = kADAPTIVE;
	s->gapstop = kGAPSTOP;
	s->warmstart = kWARMSTART;
	s->resume = kRESUME;
	s->x0 = OK_NULL;
	s->nu0 = OK_NULL;
}


pogs_solver * 
pogs_init(ok_float * A, size_t m, size_t n, 
	CBLAS_ORDER_t ord, Equilibration_t equil){

	pogs_solver * solver = OK_NULL;
	OK_TIMER t  = tic();

	/* make variables, matrix */
	POGS(pogs_solver_alloc)(&solver, m , n, ord);

	/* equilibrate A as (D * A_equil * E) = A */
	POGS(equilibrate)(solver->linalg_handle, A, solver->M, equil, ord);

	/* make projector; normalize A; adjust d, e accordingly */
	PROJECTOR(initialize)(solver->linalg_handle, solver->M->P, 1);
	POGS(normalize_DAE)(solver->linalg_handle, solver->M);	

	solver->init_time = toc(t);
	return solver;
}


void 
pogs_solve(pogs_solver * solver, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output){

	OK_TIMER t = tic();

	/* copy settings */
	POGS(update_settings)(solver->settings, settings);
	
	/* copy and scale function vectors */
	POGS(update_problem)(solver, f, g);

	/* get warm start variables */
	if (settings->warmstart)
		POGS(initialize_variables)(solver);
	if ( !(settings->resume) )
		solver->rho = settings->rho;

	info->setup_time = toc(t);
	if (!(settings->warmstart || settings->resume))
		info->setup_time += solver->init_time;
	

	/* run solver */
	t = tic();
	POGS(pogs_solver_loop)(solver, info);
	info->solve_time = toc(t);

	/* unscale output */
	POGS(copy_output)(solver, output);
}

void 
pogs_finish(pogs_solver * solver){
	POGS(pogs_solver_free)(solver);
	ok_device_reset();
}

void
pogs(ok_float * A, FunctionVector * f, FunctionVector * g,
	const pogs_settings * settings, pogs_info * info, pogs_output * output,
	CBLAS_ORDER_t ord, Equilibration_t equil){
	pogs_solver * solver = OK_NULL;
	solver = pogs_init(A, f->size, g->size, ord, equil);
	pogs_solve(solver, f, g, settings, info, output);
	pogs_finish(solver);
}


pogs_solver * 
pogs_load_solver(ok_float * A_equil, 
	ok_float * LLT_factorization, ok_float * d, 
	ok_float * e, ok_float * z, ok_float * z12, ok_float * z_dual, 
	ok_float * z_dual12, ok_float * z_prev, ok_float rho,
	size_t m, size_t n, CBLAS_ORDER_t ord){

	pogs_solver * solver = OK_NULL;
	POGS(pogs_solver_alloc)(&solver, m , n, ord);

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

void 
pogs_extract_solver(pogs_solver * solver, ok_float * A_equil, 
	ok_float * LLT_factorization, ok_float * d, 
	ok_float * e, ok_float * z, ok_float * z12, ok_float * z_dual, 
	ok_float * z_dual12, ok_float * z_prev, ok_float * rho, 
	CBLAS_ORDER_t ord){

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