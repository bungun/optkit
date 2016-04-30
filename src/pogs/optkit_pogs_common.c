#include "optkit_pogs_common.h"

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

POGS_PRIVATE ok_status block_vector_alloc(block_vector ** z, size_t m, size_t n)
{
	if (*z != OK_NULL)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_status err = OPTKIT_SUCCESS;
	block_vector * z_ = OK_NULL;
	ok_alloc(z_, sizeof(*z_));
	z_->size = m + n;
	z_->m = m;
	z_->n = n;
	ok_alloc(z_->vec, sizeof(*z_->vec));
	ok_alloc(z_->x, sizeof(*z_->x));
	ok_alloc(z_->y, sizeof(*z_->y));
	err = vector_calloc(z_->vec, m + n);
	if (!err) {
		vector_subvector(z_->y, z_->vec, 0, m);
		vector_subvector(z_->x, z_->vec, m, n);
		*z = z_;
	} else {
		OK_MAX_ERR( err, block_vector_free(z_) );
	}
	return err;
}

POGS_PRIVATE ok_status block_vector_free(block_vector * z)
{
	OK_CHECK_PTR(z);
	ok_status err = vector_free(z->vec);
	ok_free(z->x);
	ok_free(z->y);
	ok_free(z->vec);
	ok_free(z);
	return err;
}

POGS_PRIVATE ok_status pogs_variables_alloc(pogs_variables ** z, size_t m,
	size_t n)
{
	if (*z != OK_NULL)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_status err = OPTKIT_SUCCESS;
	pogs_variables * z_ = OK_NULL;
	ok_alloc(z_, sizeof(*z_));
	z_->m = m;
	z_->n = n;
	OK_CHECK_ERR( err,  block_vector_alloc(&(z_->primal), m, n) );
	OK_CHECK_ERR( err, block_vector_alloc(&(z_->primal12), m, n) );
	OK_CHECK_ERR( err, block_vector_alloc(&(z_->dual), m ,n) );
	OK_CHECK_ERR( err, block_vector_alloc(&(z_->dual12), m, n) );
	OK_CHECK_ERR( err, block_vector_alloc(&(z_->prev), m ,n) );
	OK_CHECK_ERR( err, block_vector_alloc(&(z_->temp), m, n) );
	if (err)
		OK_MAX_ERR( err, pogs_variables_free(z_) );
	else
		*z = z_;
	return err;
}

POGS_PRIVATE ok_status pogs_variables_free(pogs_variables * z)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_PTR(z);
	OK_MAX_ERR( err, block_vector_free(z->primal) );
	OK_MAX_ERR( err, block_vector_free(z->primal12) );
	OK_MAX_ERR( err, block_vector_free(z->dual) );
	OK_MAX_ERR( err, block_vector_free(z->dual12) );
	OK_MAX_ERR( err, block_vector_free(z->prev) );
	OK_MAX_ERR( err, block_vector_free(z->temp) );
	ok_free(z);
	return err;
}

POGS_PRIVATE ok_status update_settings(pogs_settings * settings,
	const pogs_settings * input)
{
	OK_CHECK_PTR(settings);
	OK_CHECK_PTR(input);
	memcpy((void *) settings, (void *) input, sizeof(pogs_settings));
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status initialize_conditions(pogs_objectives * obj,
	pogs_residuals * res, pogs_tolerances * tol,
	const pogs_settings * settings, size_t m, size_t n)
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

POGS_PRIVATE ok_status update_objective(void * linalg_handle,
	const function_vector * f, const function_vector * g, ok_float rho,
	pogs_variables * z, pogs_objectives * obj)
{
	OK_CHECK_FNVECTOR(f);
	OK_CHECK_FNVECTOR(g);
	OK_CHECK_PTR(z);
	OK_CHECK_PTR(obj);
	OK_RETURNIF_ERR( blas_dot(linalg_handle, z->primal12->vec,
		z->dual12->vec, &obj->gap) );

	ok_float val_f, val_g;

	obj->gap = MATH(fabs)(obj->gap);

	OK_RETURNIF_ERR( function_eval_vector(f, z->primal12->y, &val_f) );
	OK_RETURNIF_ERR( function_eval_vector(g, z->primal12->x, &val_g) );

	obj->primal = val_f + val_g;
	obj->dual = obj->primal - obj->gap;
	return OPTKIT_SUCCESS;
}

/*
 * eps_primal = eps_abs + eps_rel * sqrt(m) * ||y^k+1/2||
 * eps_dual = eps_abs + eps_rel * sqrt(n) * ||xt^k+1/2||
 * eps_gap = eps_abs + eps_rel * sqrt(mn) * ||z^k|| * ||z^k+1/2||
 */
POGS_PRIVATE ok_status update_tolerances(void * linalg_handle,
	pogs_variables * z, pogs_objectives * obj, pogs_tolerances * eps)
{
	ok_float nrm_z, nrm_z12;
	OK_CHECK_PTR(z);
	OK_CHECK_PTR(obj);
	OK_CHECK_PTR(eps);

	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->primal12->y, &nrm_z12) );
	eps->primal = eps->atolm + eps->reltol * nrm_z12;

	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->dual12->x, &nrm_z12) );
	eps->dual = eps->atoln + eps->reltol * nrm_z12;

	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->primal->vec, &nrm_z) );
	OK_RETURNIF_ERR( blas_nrm2(linalg_handle, z->primal12->vec, &nrm_z12) );
	eps->gap = eps->atolmn + eps->reltol * MATH(sqrt)(nrm_z) *
		MATH(sqrt)(nrm_z12);

	return OPTKIT_SUCCESS;
}

/* z^k <- z^{k+1} */
POGS_PRIVATE ok_status set_prev(pogs_variables * z)
{
	return OK_SCAN_ERR( vector_memcpy_vv(z->prev->vec, z->primal->vec) );
}

/*
 * update z^{k + 1/2} according to following rule:
 *
 *	y^{k+1/2} = Prox_{rho, f} (y^k - yt^k)
 *	x^{k+1/2} = Prox_{rho, g} (x^k - xt^k)
 */
POGS_PRIVATE ok_status prox(void * linalg_handle, const function_vector * f,
	const function_vector * g, pogs_variables * z, ok_float rho)
{
	OK_RETURNIF_ERR(
		vector_memcpy_vv(z->temp->vec, z->primal->vec) );
	OK_RETURNIF_ERR(
		blas_axpy(linalg_handle, -kOne, z->dual->vec, z->temp->vec) );
	OK_RETURNIF_ERR(
		prox_eval_vector(f, rho, z->temp->y, z->primal12->y) );
	return OK_SCAN_ERR(
		prox_eval_vector(g, rho, z->temp->x, z->primal12->x));
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
	OK_CHECK_PTR(z);
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
POGS_PRIVATE ok_status adaptrho(pogs_variables * z,
	const pogs_settings * settings, ok_float * rho, adapt_params * params,
	const pogs_residuals * res, const pogs_tolerances * eps, const uint k)
{
	if (!(settings->adaptiverho))
		return OPTKIT_SUCCESS;

	if (!z || !rho || !params || !res || !eps)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	if (res->dual < params->xi * eps->dual &&
		res->primal > params->xi * eps->primal &&
		kTAU * (ok_float) k > params->l) {

		if (*rho < kRHOMAX) {
			*rho *= params->delta;
			if (settings->verbose > 2)
				printf("+RHO: %.3e\n", *rho);
			vector_scale(z->dual->vec, kOne / params->delta);
			params->delta = (params->delta * kGAMMA < kDELTAMAX) ?
				params->delta * kGAMMA : kDELTAMAX;
			params->u = (ok_float) k;
		}
	} else if (res->dual > params->xi * eps->dual &&
			res->primal < params->xi * eps->primal &&
			kTAU * (ok_float) k > (ok_float) params->u) {

		if (*rho > kRHOMIN) {
			*rho /= params->delta;
			if (settings->verbose > 2)
				printf("-RHO: %.3e\n", *rho);

			vector_scale(z->dual->vec, params->delta);
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
POGS_PRIVATE ok_status copy_output(pogs_variables * z, vector * d, vector * e,
	pogs_output * output, const ok_float rho, const uint suppress)
{
	OK_CHECK_PTR(z);
	OK_CHECK_VECTOR(d);
	OK_CHECK_VECTOR(e);
	if (z->n != e->size || z->m != d->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	vector_memcpy_vv(z->temp->vec, z->primal12->vec);
	vector_mul(z->temp->x, e);
	vector_memcpy_av(output->x, z->temp->x, 1);

	if (suppress < 2) {
		vector_div(z->temp->y, d);
		vector_memcpy_av(output->y, z->temp->y, 1);
	}

	if (suppress < 3) {
		vector_memcpy_vv(z->temp->vec, z->dual12->vec);
		vector_scale(z->temp->vec, -rho);
		vector_mul(z->temp->y, d);
		vector_memcpy_av(output->nu, z->temp->y, 1);

		if (suppress < 1) {
			vector_div(z->temp->x, e);
			vector_memcpy_av(output->mu, z->temp->x, 1);
		}
	}
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status print_header_string()
{
	printf("\n   #    %s    %s    %s   %s   %s        %s    %s\n",
		"res_pri", "eps_pri", "res_dual", "eps_dual",
		"gap", "eps_gap", "objective");
	printf("   -    %s    %s    %s   %s   %s        %s    %s\n",
		"-------", "-------", "--------", "--------",
		"---", "-------", "---------");
	return OPTKIT_SUCCESS;
}

POGS_PRIVATE ok_status print_iter_string(pogs_residuals * res,
	pogs_tolerances * eps, pogs_objectives * obj, uint k)
{
	OK_CHECK_PTR(res);
	OK_CHECK_PTR(eps);
	OK_CHECK_PTR(obj);
	printf("   %u: %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e, %0.3e\n",
		k, res->primal, eps->primal, res->dual, eps->dual,
		res->gap, eps->gap, obj->primal);
	return OPTKIT_SUCCESS;
}

ok_status set_default_settings(pogs_settings * s)
{
	OK_CHECK_PTR(s);
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
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
