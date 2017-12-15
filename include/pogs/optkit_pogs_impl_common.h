#ifndef OPTKIT_POGS_IMPL_COMMON_H_
#define OPTKIT_POGS_IMPL_COMMON_H_

#include "optkit_pogs_datatypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* FORWARD DECLARATIONS */
ok_status pogs_graph_vector_alloc(pogs_graph_vector *z, size_t m, size_t n);
ok_status pogs_graph_vector_free(pogs_graph_vector *z);
ok_status pogs_graph_vector_attach_memory(pogs_graph_vector *z);
ok_status pogs_graph_vector_release_memory(pogs_graph_vector *z);
ok_status pogs_graph_vector_view_vector(pogs_graph_vector *z, vector *v,
	size_t offset, size_t length);

ok_status pogs_variables_alloc(pogs_variables *z, size_t m, size_t n);
ok_status pogs_variables_free(pogs_variables *z);

ok_status pogs_set_default_settings(pogs_settings *s);
ok_status pogs_update_settings(pogs_settings *settings,
	const pogs_settings *input);

ok_status pogs_initialize_conditions(pogs_objective_values *obj,
	pogs_residuals *res, pogs_tolerances *tol,
	const pogs_settings *settings, size_t m, size_t n);

ok_status pogs_update_objective_values(void *linalg_handle,
	const function_vector *f, const function_vector *g, ok_float rho,
	pogs_variables *z, pogs_objective_values *obj);

ok_status pogs_update_tolerances(void *linalg_handle,
	pogs_variables *z, pogs_objective_values *obj,
	pogs_tolerances *tol);

ok_status pogs_set_print_iter(uint *PRINT_ITER, const pogs_settings *settings);
ok_status pogs_print_header_string(void);
ok_status pogs_print_iter_string(pogs_residuals *res,
	pogs_tolerances *tol, pogs_objective_values *obj, uint k);

ok_status pogs_scale_objectives(function_vector *f_solver,
	function_vector *g_solver, vector *d, vector *e,
	const function_vector *f, const function_vector *g);
ok_status pogs_unscale_output(pogs_output *output,
	const pogs_variables *z, const vector *d, const vector *e,
	const ok_float rho, const uint suppress);

/* DEFINITIONS */
ok_status pogs_graph_vector_alloc(pogs_graph_vector *z, size_t m, size_t n)
{
	OK_CHECK_PTR(z);
	if (z->vec)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_status err = OPTKIT_SUCCESS;
	z->size = m + n;
	z->m = m;
	z->n = n;
	ok_alloc(z->vec, sizeof(*z->vec));
	ok_alloc(z->x, sizeof(*z->x));
	ok_alloc(z->y, sizeof(*z->y));
	z->memory_attached = 0;
	return err;
}

ok_status pogs_graph_vector_free(pogs_graph_vector *z)
{
	OK_CHECK_PTR(z);
	ok_status err = OPTKIT_SUCCESS;
	if (z->memory_attached)
		err = OK_SCAN_ERR( pogs_graph_vector_release_memory(z) );
	ok_free(z->x);
	ok_free(z->y);
	ok_free(z->vec);
	return err;
}
ok_status pogs_graph_vector_attach_memory(pogs_graph_vector *z)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (z->vec->data != OK_NULL)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	err = OK_SCAN_ERR( vector_calloc(z->vec, z->m + z->n) );
	OK_CHECK_ERR( err, vector_subvector(z->y, z->vec, 0, z->m) );
	OK_CHECK_ERR( err, vector_subvector(z->x, z->vec, z->m, z->n) );
	z->memory_attached = 1;
	if (err)
		OK_MAX_ERR( err, pogs_graph_vector_release_memory(z) );
	return err;
}

ok_status pogs_graph_vector_release_memory(pogs_graph_vector *z)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!z)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	err = OK_SCAN_ERR( vector_free(z->vec) );
	z->y->data = OK_NULL;
	z->x->data = OK_NULL;
	z->memory_attached = 0;
	return err;
}

ok_status pogs_graph_vector_view_vector(pogs_graph_vector *z, vector *v,
	size_t offset, size_t length)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_VECTOR(v);

	if (z->vec->data != OK_NULL)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	if (offset + length > v->size)
		err = OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_CHECK_ERR( err, vector_subvector(z->vec, v, offset, length) );
	OK_CHECK_ERR( err, vector_subvector(z->y, z->vec, 0, z->m) );
	OK_CHECK_ERR( err, vector_subvector(z->x, z->vec, z->m, z->n) );
	return err;
}

ok_status pogs_graph_vector_copy(pogs_graph_vector *z_target,
	const pogs_graph_vector *z_source)
{
	if (!z_target || !z_source)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	return OK_SCAN_ERR( vector_memcpy_vv(z_target->vec, z_source->vec) );
}
ok_status pogs_variables_alloc(pogs_variables *z, size_t m, size_t n)
{
	OK_CHECK_PTR(z);
	if (z->state)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_status err = OPTKIT_SUCCESS;
	z->m = m;
	z->n = n;

	ok_alloc(z->state, sizeof(*(z->state)));
	OK_CHECK_ERR( err, vector_calloc(z->state, 6 * (m + n)) );

	ok_alloc(z->primal, sizeof(*z));
	OK_CHECK_ERR( err, pogs_graph_vector_alloc(z->primal, m, n) );
	OK_CHECK_ERR( err, pogs_graph_vector_view_vector(z->primal, z->state,
		0, m + n) );
	ok_alloc(z->dual, sizeof(*z));
	OK_CHECK_ERR( err, pogs_graph_vector_alloc(z->dual, m, n) );
	OK_CHECK_ERR( err, pogs_graph_vector_view_vector(z->dual, z->state,
		m + n, m + n) );
	ok_alloc(z->primal12, sizeof(*z));
	OK_CHECK_ERR( err, pogs_graph_vector_alloc(z->primal12, m, n) );
	OK_CHECK_ERR( err, pogs_graph_vector_view_vector(z->primal12, z->state,
		2 * (m + n), m + n) );
	ok_alloc(z->dual12, sizeof(*z));
	OK_CHECK_ERR( err, pogs_graph_vector_alloc(z->dual12, m, n) );
	OK_CHECK_ERR( err, pogs_graph_vector_view_vector(z->dual12, z->state,
		3 * (m + n), m + n) );
	ok_alloc(z->prev, sizeof(*z));
	OK_CHECK_ERR( err, pogs_graph_vector_alloc(z->prev, m ,n) );
	OK_CHECK_ERR( err, pogs_graph_vector_view_vector(z->prev, z->state,
		4 * (m + n), m + n) );
	ok_alloc(z->temp, sizeof(*z));
	OK_CHECK_ERR( err, pogs_graph_vector_alloc(z->temp, m, n) );
	OK_CHECK_ERR( err, pogs_graph_vector_view_vector(z->temp, z->state,
		5 * (m + n), m + n) );

	ok_alloc(z->fixed_point_iterate, sizeof(*(z->fixed_point_iterate)));
	OK_CHECK_ERR( err, vector_subvector(z->fixed_point_iterate, z->state,
		0, 2 * (m + n)) );

	if (err)
		OK_MAX_ERR( err, pogs_variables_free(z) );
	return err;
}

ok_status pogs_variables_free(pogs_variables *z)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(z);
	OK_MAX_ERR( err, pogs_graph_vector_free(z->primal) );
	ok_free(z->primal);
	OK_MAX_ERR( err, pogs_graph_vector_free(z->primal12) );
	ok_free(z->primal12);
	OK_MAX_ERR( err, pogs_graph_vector_free(z->dual) );
	ok_free(z->dual);
	OK_MAX_ERR( err, pogs_graph_vector_free(z->dual12) );
	ok_free(z->dual12);
	OK_MAX_ERR( err, pogs_graph_vector_free(z->prev) );
	ok_free(z->prev);
	OK_MAX_ERR( err, pogs_graph_vector_free(z->temp) );
	ok_free(z->temp);
	OK_MAX_ERR( err, vector_free(z->state) );
	ok_free(z->fixed_point_iterate);
	ok_free(z->state);
	return err;
}

ok_status pogs_set_default_settings(pogs_settings *s)
{
	OK_CHECK_PTR(s);
	s->alpha = kALPHA;
	s->rho = kOne;
	s->abstol = kATOL;
	s->reltol = kRTOL;
	s->tolproj = kTOLPROJ;
	s->toladapt = kTOLADAPT;
	s->anderson_regularization = kMU;
	s->maxiter = kMAXITER;
	s->anderson_lookback = kANDERSON;
	s->verbose = kVERBOSE;
	s->suppress = kSUPPRESS;
	s->adaptiverho = kADAPTIVE;
	s->accelerate = kACCELERATE;
	s->gapstop = kGAPSTOP;
	s->warmstart = kWARMSTART;
	s->resume = kRESUME;
	s->diagnostic = kDIAGNOSTIC;
	s->x0 = OK_NULL;
	s->nu0 = OK_NULL;
	return OPTKIT_SUCCESS;
}

ok_status pogs_update_settings(pogs_settings *settings,
	const pogs_settings *input)
{
	OK_CHECK_PTR(settings);
	OK_CHECK_PTR(input);
	settings->alpha = input->alpha;
	settings->rho = input->rho;
	settings->abstol = input->abstol;
	settings->reltol = input->reltol;
	settings->tolproj = input->tolproj;
	settings->toladapt = input->toladapt;
	settings->anderson_regularization = input->anderson_regularization;
	settings->maxiter = input->maxiter;
	settings->anderson_lookback = input->anderson_lookback;
	settings->verbose = input->verbose;
	settings->suppress = input->suppress;
	settings->adaptiverho = input->adaptiverho;
	settings->accelerate = input->accelerate;
	settings->gapstop = input->gapstop;
	settings->warmstart = input->warmstart;
	settings->resume = input->resume;
	settings->diagnostic = input->diagnostic;
	settings->x0 = input->x0;
	settings->nu0 = input->nu0;
	return OPTKIT_SUCCESS;
}

ok_status pogs_initialize_conditions(pogs_objective_values *obj,
	pogs_residuals *res, pogs_tolerances *tol,
	const pogs_settings *settings, size_t m, size_t n)
{
	if (!obj || !res || !tol || !settings)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	obj->primal = OK_NAN;
	obj->dual = OK_NAN;
	obj->gap = OK_NAN;
	res->primal = OK_NAN;
	res->dual = OK_NAN;
	res->gap = OK_NAN;
	tol->primal = kZero;
	tol->dual = kZero;
	tol->gap = kZero;
	tol->reltol = settings->reltol;
	tol->abstol = settings->abstol;
	tol->atolm = MATH(sqrt)((ok_float) m) * settings->abstol;
	tol->atoln = MATH(sqrt)((ok_float) n) * settings->abstol;
	tol->atolmn = MATH(sqrt)((ok_float) (m * n)) * settings->abstol;
	return OPTKIT_SUCCESS;
}

ok_status pogs_update_objective_values(void *linalg_handle,
	const function_vector *f, const function_vector *g, ok_float rho,
	pogs_variables *z, pogs_objective_values *obj)
{
	ok_float val_f, val_g;

	OK_CHECK_FNVECTOR(f);
	OK_CHECK_FNVECTOR(g);
	if (!z || !obj)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	OK_RETURNIF_ERR( blas_dot(linalg_handle, z->primal12->vec,
		z->dual12->vec, &obj->gap) );

	obj->gap = MATH(fabs)(obj->gap);

	OK_RETURNIF_ERR( function_eval_vector(f, z->primal12->y, &val_f) );
	OK_RETURNIF_ERR( function_eval_vector(g, z->primal12->x, &val_g) );

	obj->primal = val_f + val_g;
	obj->dual = obj->primal - obj->gap;
	return OPTKIT_SUCCESS;
}

/*
 * tol_primal = tol_abs + tol_rel * sqrt(m) * ||y^k+1/2||
 * tol_dual = tol_abs + tol_rel * sqrt(n) * ||xt^k+1/2||
 * tol_gap = tol_abs + tol_rel * sqrt(mn) * ||z^k|| * ||z^k+1/2||
 */
ok_status pogs_update_tolerances(void *linalg_handle,
	pogs_variables *z, pogs_objective_values *obj, pogs_tolerances *tol)
{
	ok_float nrm_z, nrm_z12;
	if (!z || !obj || !tol)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->primal12->y, &nrm_z12) );
	tol->primal = tol->atolm + tol->reltol * nrm_z12;

	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->dual12->x, &nrm_z12) );
	tol->dual = tol->atoln + tol->reltol * nrm_z12;

	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->primal->vec, &nrm_z) );
	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->primal12->vec, &nrm_z12) );
	tol->gap = tol->atolmn + tol->reltol * MATH(sqrt)(nrm_z) *
		MATH(sqrt)(nrm_z12);

	return OPTKIT_SUCCESS;
}

ok_status pogs_set_print_iter(uint *PRINT_ITER, const pogs_settings *settings)
{
	uint iter = 10000u, k;

	if (!settings || !PRINT_ITER)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

 	if (settings->verbose == 0)
		iter = settings->maxiter * 2u;
	else
		for (k = 0; k < settings->verbose && iter > 1; ++k)
			iter /= 10;
	*PRINT_ITER = iter;
	return OPTKIT_SUCCESS;
}

ok_status pogs_print_header_string(void)
{
	printf("\n");
	printf("%8s %12s %12s %12s %12s %12s %12s %12s\n",
		"iter #", "res_pri", "eps_pri", "res_dual", "eps_dual", "gap",
		"eps_gap", "objective");
	printf("%8s %12s %12s %12s %12s %12s %12s %12s\n",
		"------", "-------", "-------", "--------", "--------", "---",
		"-------", "---------");
	return OPTKIT_SUCCESS;
}

ok_status pogs_print_iter_string(pogs_residuals *res,
	pogs_tolerances *tol, pogs_objective_values *obj, uint k)
{
	OK_CHECK_PTR(res);
	OK_CHECK_PTR(tol);
	OK_CHECK_PTR(obj);
	printf("%8u %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e\n",
		k, res->primal, tol->primal, res->dual, tol->dual,
		res->gap, tol->gap, obj->primal);
	return OPTKIT_SUCCESS;
}


ok_status pogs_scale_objectives(function_vector *f_solver,
	function_vector *g_solver, vector *d, vector *e,
	const function_vector *f, const function_vector *g)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_FNVECTOR(f_solver);
	OK_CHECK_FNVECTOR(g_solver);
	OK_CHECK_VECTOR(d);
	OK_CHECK_VECTOR(e);
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_FNVECTOR(g);

	OK_CHECK_ERR( err, function_vector_memcpy_va(f_solver, f->objectives) );
	OK_CHECK_ERR( err, function_vector_memcpy_va(g_solver, g->objectives) );
	OK_CHECK_ERR( err, function_vector_div(f_solver, d) );
	OK_CHECK_ERR( err, function_vector_mul(g_solver, e) );
	return err;
}

/*
 * copy pogs variables to outputs:
 *
 * 	Ex^{k + 1/2} --> output.x
 * 	D^{-1}y^{k + 1/2} --> output.y
 * 	-rho * Ext^{k + 1/2} --> output.mu
 * 	-rho * D^{-1}yt^{k + 1/2} --> output.nu
 *
 *
 * output suppression levels:
 *
 *	0: copy (x, y, mu, nu), suppress ()
 *	1: copy (x, y, nu), suppress (mu)
 *	2: copy (x, nu), suppress (y, mu)
 *	3: copy (x), suppress (y, mu, nu)
 */
ok_status pogs_unscale_output(pogs_output *output,
	const pogs_variables *z, const vector *d, const vector *e,
	const ok_float rho, const uint suppress)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(z);
	OK_CHECK_VECTOR(d);
	OK_CHECK_VECTOR(e);
	if (z->n != e->size || z->m != d->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_CHECK_ERR( err, pogs_graph_vector_copy(z->temp, z->primal12) );
	OK_CHECK_ERR( err, vector_mul(z->temp->x, e) );
	OK_CHECK_ERR( err, vector_memcpy_av(output->x, z->temp->x, 1) );

	if (suppress < 2) {
		OK_CHECK_ERR( err, vector_div(z->temp->y, d) );
		OK_CHECK_ERR( err, vector_memcpy_av(output->y, z->temp->y, 1) );
	}

	if (suppress < 3) {
		OK_CHECK_ERR( err, pogs_graph_vector_copy(z->temp, z->dual12) );
		OK_CHECK_ERR( err, vector_scale(z->temp->vec, -rho) );
		OK_CHECK_ERR( err, vector_mul(z->temp->y, d) );
		OK_CHECK_ERR( err, vector_memcpy_av(output->nu, z->temp->y, 1) );
		if (suppress < 1) {
			OK_CHECK_ERR( err, vector_div(z->temp->x, e) );
			OK_CHECK_ERR( err, vector_memcpy_av(output->mu,
				z->temp->x, 1) );
		}
	}
	return err;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_POGS_IMPL_COMMON_H_ */
