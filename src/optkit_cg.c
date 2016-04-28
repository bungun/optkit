/*
 * The CGLS routine below is adapted from Chris Fougner's software package POGS
 * (original license below)
 *
 * The preconditioned CG routine below is adapted from Brendan O'Donoghue's
 * software package SCS
 * (original license below)
 *
 *
 *
 * original license for cgls.h and cgls.cuh in POGS:
 * =================================================
 *
 * Copyright (c) 2015, Christopher Fougner
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *
 * original license for SCS
 * ========================
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2012 Brendan O'Donoghue (bodonoghue85@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "optkit_cg.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FLOAT
const ok_float kEps = (ok_float) 1e-8;
#else
const ok_float kEps = (ok_float) 1e-16;
#endif

cgls_helper * cgls_helper_alloc(size_t m, size_t n)
{
	ok_status err = OPTKIT_SUCCESS;
	cgls_helper * h;
	h = malloc(sizeof(*h));
	memset(h, 0, sizeof(*h));
	h->norm_s = (ok_float) 0;
	h->norm_s0 = (ok_float) 0;
	h->norm_x = (ok_float) 0;
	h->xmax = (ok_float) 0;
	h->alpha = (ok_float) 0;
	h->beta = (ok_float) 0;
	h->delta = (ok_float) 0;
	h->gamma = (ok_float) 0;
	h->gamma_prev = (ok_float) 0;
	h->shrink = (ok_float) 0;
	err = vector_calloc(&(h->p), n);
	OK_CHECK_ERR( err, vector_calloc(&(h->q), m) );
	OK_CHECK_ERR( err, vector_calloc(&(h->r), m) );
	OK_CHECK_ERR( err, vector_calloc(&(h->s), n) );
	OK_CHECK_ERR( err, blas_make_handle(&(h->blas_handle)) );
	if (!err)
		h->indicator = (int *) malloc(sizeof(int));
	return h;
}

ok_status cgls_helper_free(cgls_helper * helper)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!helper || !helper->p.data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (helper->blas_handle)
	 	err = blas_destroy_handle(helper->blas_handle);
	if (helper->p.data)
		OK_MAX_ERR( err, vector_free(&(helper->p)) );
	if (helper->q.data)
		OK_MAX_ERR( err, vector_free(&(helper->q)) );
	if (helper->r.data)
		OK_MAX_ERR( err, vector_free(&(helper->r)) );
	if (helper->s.data)
		OK_MAX_ERR( err, vector_free(&(helper->s)) );
	if (helper->indicator)
		ok_free(helper->indicator);
	ok_free(helper);
	return err;
}

pcg_helper * pcg_helper_alloc(size_t m, size_t n)
{
	ok_status err = OPTKIT_SUCCESS;
	pcg_helper * h;
	h = malloc(sizeof(*h));
	memset(h, 0, sizeof(*h));
	h->norm_r = (ok_float) 0;
	h->alpha = (ok_float) 0;
	h->gamma = (ok_float) 0;
	h->gamma_prev = (ok_float) 0;
	OK_CHECK_ERR( err, vector_calloc(&(h->p), n) );
	OK_CHECK_ERR( err, vector_calloc(&(h->q), n) );
	OK_CHECK_ERR( err, vector_calloc(&(h->r), n) );
	OK_CHECK_ERR( err, vector_calloc(&(h->z), n) );
	OK_CHECK_ERR( err, vector_calloc(&(h->temp), m) );
	OK_CHECK_ERR( err, blas_make_handle(&(h->blas_handle)) );
	h->never_solved = 1;
	if (!err)
		h->indicator = (int *) malloc(sizeof(int));
	return h;
}

ok_status pcg_helper_free(pcg_helper * helper)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!helper || !helper->p.data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	if (helper->blas_handle)
		err = blas_destroy_handle(helper->blas_handle);
	if (helper->p.data)
		OK_MAX_ERR( err, vector_free(&(helper->p)) );
	if (helper->q.data)
		OK_MAX_ERR( err, vector_free(&(helper->q)) );
	if (helper->r.data)
		OK_MAX_ERR( err, vector_free(&(helper->r)) );
	if (helper->z.data)
		OK_MAX_ERR( err, vector_free(&(helper->z)) );
	if (helper->temp.data)
		OK_MAX_ERR( err, vector_free(&(helper->temp)) );
	if (helper->indicator)
		ok_free(helper->indicator);
	ok_free(helper);
	return err;
}

/*
 *  CGLS Conjugate Gradient Least squares
 *
 *  Attempts to solve the least squares problem
 *
 *    min. ||Ax - b||_2^2 + rho ||x||_2^2
 *
 *  using the Conjugate Gradient for Least Squares method.
 *  This is more stable than applying CG to the normal equations.
 *
 * ref: Chris Fougner, POGS
 * https://github.com/foges/pogs/blob/master/src/cpu/include/cgls.h
 */
ok_status cgls_nonallocating(cgls_helper * helper,
		operator * op, vector * b, vector * x,
		const ok_float rho, const ok_float tol,
		const size_t maxiter, const int quiet,
		uint * flag)
{
	if (!helper || !helper->indicator)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_OPERATOR(op);
	OK_CHECK_VECTOR(b);
	OK_CHECK_VECTOR(x);
	OK_CHECK_PTR(flag);

	/* convenience abbreviations */
	cgls_helper * h = helper;
	vector p = h->p;
	vector q = h->q;
	vector r = h->r;
	vector s = h->s;
	void * blas_hdl = h->blas_handle;

	/* variable and constant declarations */
	char fmt[] = "%5d %9.2e %12.5e\n";
	uint k;
	ok_float p_squared;
	int indefinite = 0, converged = 0;
	const ok_float kNegRho = -rho;

	/* dimension check */
	if (op->size1 != b->size || op->size2 != x->size || r.size != b->size
		|| s.size != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );



	/* initialization */
	blas_nrm2(blas_hdl, x, &h->norm_x);

	/* r = b - Ax */
	vector_memcpy_vv(&r, b);
	if (h->norm_x > 0)
		op->fused_apply(op->data, -kOne, x, kOne, &r);

	/* s = A'*r - rho * x */
	op->adjoint(op->data, &r, &s);
	if (h->norm_x > 0)
		blas_axpy(blas_hdl, kNegRho, x, &s);

	/* p = s */
	vector_memcpy_vv(&p, &s);

	/* initial norms (norm_x calculated above) */
	blas_nrm2(blas_hdl, &s, &h->norm_s);
	h->norm_s0 =  h->norm_s;
	h->gamma = h->norm_s0 * h->norm_s0;
	h->xmax = h->norm_x;

	*flag = 0;
	if (h->norm_s < kEps)
		*flag = 1;

	if (!quiet && !*flag)
		printf("    k    norm x        resNE\n");

	/* ------------------- CGLS ----------------- */
	for (k = 0; k < maxiter && !*flag; ++k) {
		/* q = Ap */
		op->apply(op->data, &p, &q);

		/*
		 * NOTE: CHRIS' COMMENT SAYS
		 * delta = ||p||_2^2 + rho * ||q||_2^2
		 * BUT THE CODE PERFORMS:
		 * delta = ||q||_2^2 + rho * ||p||_2^2
		 */
		blas_dot(blas_hdl, &q, &q, &h->delta);
		blas_dot(blas_hdl, &p, &p, &p_squared);
		h->delta += rho * p_squared;

		if (h->delta <= 0)
			indefinite = 1;
		if (h->delta == 0)
			h->delta = kEps;

		h->alpha = h->gamma / h->delta;
		/* x += alpha * p */
		/* r -= alpha * q */
		blas_axpy(blas_hdl, h->alpha, &p, x);
		blas_axpy(blas_hdl, -(h->alpha), &q, &r);

		/* s = A'r - rho * x */
		op->adjoint(op->data, &r, &s);
		blas_axpy(blas_hdl, kNegRho, x, &s);

		/* compute gamma = ||s||^2 */
		/* compute beta = gamma/gamma_prev */
		blas_nrm2(blas_hdl, &s, &h->norm_s);
		h->gamma_prev = h->gamma;
		h->gamma = h->norm_s * h->norm_s;
		h->beta = h->gamma / h->gamma_prev;

		/* p = s + beta * p */
		vector_scale(&p, h->beta);
		blas_axpy(blas_hdl, kOne, &s, &p);

		/* convergence check */
		blas_nrm2(blas_hdl, x, &h->norm_x);
		h->xmax = (h->norm_x > h->xmax) ? h->norm_x : h->xmax;
		converged = ((h->norm_s < h->norm_s0 * tol) ||
			(h->norm_x * tol > 1));
		if (!quiet && (converged || (k + 1) % 10 == 0))
			printf(fmt, k + 1, h->norm_x, h->norm_s / h->norm_s0);
		if (converged)
			break;
	}

	/* determine exit status */
	h->shrink = h->norm_x / h->xmax;
	if (k == maxiter)
		*flag = 2;
	else if (indefinite)
		*flag = 3;
	else if (h->shrink * h->shrink <= tol)
		*flag = 4;

	return OPTKIT_SUCCESS;
}

uint cgls(operator * op, vector * b, vector * x, const ok_float rho,
	const ok_float tol, const size_t maxiter, const int quiet)
{
	uint flag;
	ok_status err = OPTKIT_SUCCESS;

	cgls_helper * helper = cgls_helper_alloc(op->size1, op->size2);

	/* CGLS call */
	err = cgls_nonallocating(helper, op, b, x, rho, tol, maxiter, quiet,
		&flag);
	if (err)
		flag = INT_MAX;

	cgls_helper_free(helper);

	return flag;
}

void * cgls_init(size_t m, size_t n)
{
	return (void *) cgls_helper_alloc(m, n);
}

uint cgls_solve(void * cgls_work, operator * op, vector * b, vector * x,
	const ok_float rho, const ok_float tol, const size_t maxiter, int quiet)
{
	uint flag;
	if ( cgls_nonallocating((cgls_helper *) cgls_work, op, b, x, rho,
		tol, maxiter, quiet, &flag) )
		return INT_MAX;
	else
		return flag;
}

ok_status cgls_finish(void * cgls_work)
{
	return cgls_helper_free((cgls_helper *) cgls_work);
}

/*
 * diagonal preconditioner
 *
 * need column norms-squared of
 * 	(rho I + A'A)
 *
 * to obtain this:
 *
 *
 *
 * Ae_i = a_i
 * A'Ae_i = A'a_i = (aa)_i
 * (aa)_i + rho * e_i : column
 */
ok_status diagonal_preconditioner(operator * op, vector * p, ok_float rho)
{
	OK_CHECK_OPERATOR(op);
	OK_CHECK_VECTOR(p);

	ok_status err = OPTKIT_SUCCESS;
	ok_float col_norm_sq;
	size_t i;
	void * blas_handle;
	vector ej, ej_sub, a, iaa, p_sub;
	ej.data = OK_NULL;
	ej_sub.data = OK_NULL;
	a.data = OK_NULL;
	iaa.data = OK_NULL;
	p_sub.data = OK_NULL;

	if (p->size != op->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	vector_calloc(&ej, op->size2);
	vector_calloc(&a, op->size1);
	vector_calloc(&iaa, op->size2);
	blas_make_handle(&blas_handle);

	vector_scale(p, 0);

	printf("%s %f\n", "RHO C", rho);
	for (i = 0; i < op->size2 && !err; ++i) {
		vector_scale(&ej, kZero);
		vector_subvector(&ej_sub, &ej, i, 1);
		vector_add_constant(&ej_sub, 1);
		op->apply(op->data, &ej, &a);
		op->adjoint(op->data, &a, &iaa);

		blas_dot(blas_handle, &iaa, &iaa, &col_norm_sq);
		vector_subvector(&p_sub, p, i, 1);
		vector_add_constant(&p_sub, col_norm_sq + rho);
	}

	err = vector_recip(p);

	vector_free(&ej);
	vector_free(&a);
	vector_free(&iaa);
	blas_destroy_handle(blas_handle);
	return err;
}

/*
 * Preconditioned Conjugate Gradient (solve)
 *
 * Attempts to solve
 *
 * 		M(rho * I + A'A)x = b
 *
 * to specified tolerance within maxiter CG iterations
 *
 * Solution written to method parameter x, and stored in helper->z
 * Warm starts from helper->z if helper->never_ran is false
 *
 * ref: Brendan O'Donoghue, SCS
 * https://github.com/cvxgrp/scs/blob/master/linsys/indirect/private.c
 */
ok_status pcg_nonallocating(pcg_helper * helper, operator * op,
	operator * pre_cond, vector * b, vector * x, const ok_float rho,
	const ok_float tol, const size_t maxiter, const int quiet, uint * iters)
{
	if (!helper || !helper->indicator)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_OPERATOR(op);
	OK_CHECK_OPERATOR(pre_cond);
	OK_CHECK_VECTOR(b);
	OK_CHECK_VECTOR(x);
	OK_CHECK_PTR(iters);

	ok_status err = OPTKIT_SUCCESS;
	/*
	 * convenience abbreviations
	 * p, q: iterate vectors
	 * r: residual
	 * z: preconditioned variable
	 * x0: alias of z for warmstart
	 * temp: storage for A'A intermediate
	 */
	pcg_helper * h = helper;
	vector p = h->p;
	vector q = h->q;
	vector r = h->r;
	vector z = h->z;
	vector x0 = h->z;
	vector temp = h->temp;
	void * blas_hdl = h->blas_handle;

	/* variable/constant declarations */
	uint k;
	const ok_float kNormTol = (tol < 1e-18) ? tol : (ok_float) 1e-18;
	char fmt [] = "tol: %.4e, resid: %.4e, iters: %u\n";
	char nocvg_fmt [] = "did not converge in %u iters (max iter = %u)\n";

	/* dimension checks */
	if (pre_cond->size1 != pre_cond->size2 || pre_cond->size1 != op->size2
		|| op->size2 != x->size || x->size != b->size ||
		x->size != p.size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	/* initialization */
	if (h->never_solved) {
		/* r = b */
		/* x = 0 */
		vector_memcpy_vv(&r, b);
		vector_scale(x, kZero);
	} else {
		/* r = b - (rho * I + A'A)x0 */
		/* x = x0 */
		vector_memcpy_vv(&r, b);
		op->apply(op->data, &x0, &temp);
		op->fused_adjoint(op->data, -kOne, &temp, kOne, &r);
		blas_axpy(blas_hdl, -rho, &x0, &r);
		vector_memcpy_vv(x, &x0);
	}

	/* check to see if we need to run CG at all */
	blas_dot(blas_hdl, &r, &r, &h->norm_r);
	h->norm_r = MATH(sqrt)(h->norm_r);
	if (h->norm_r < kNormTol)
	        return 0;

	/* p = Mr (apply preconditioner) */
	pre_cond->apply(pre_cond->data, &r, &p);
	/* gamma = r'Mr */
	blas_dot(blas_hdl, &r, &p, &h->gamma);

	for (k = 0; k < maxiter && !err; ++k) {
		/* q = (rho * I + A'A)p */
		op->apply(op->data, &p, &temp);
		op->adjoint(op->data, &temp, &q);
		blas_axpy(blas_hdl, rho, &p, &q);

		/* alpha = <p, r> / <p, q> */
		blas_dot(blas_hdl, &p, &q, &h->alpha);
		if (!err)
			h->alpha = h->gamma / h->alpha;

		/* x += alpha * p */
		/* r -= alpha * q */
		blas_axpy(blas_hdl, +h->alpha, &p, x);
		blas_axpy(blas_hdl, -h->alpha, &q, &r);

		/* check convergence */
		blas_dot(blas_hdl, &r, &r, &h->norm_r);
		if (!err)
			h->norm_r = MATH(sqrt)(h->norm_r);
		if (h->norm_r <= tol) {
			k += 1;
			if (!quiet)
				printf(fmt, tol, h->norm_r, k);
			break;
		}

		h->gamma_prev = h->gamma;

		/* z = Mr */
		pre_cond->apply(pre_cond->data, &r, &z);

		/* gamma = r'Mr */
		blas_dot(blas_hdl, &r, &z, &h->gamma);

		/* p = p * gamma / gamma_prev + Mr */
		vector_scale(&p, h->gamma / h->gamma_prev);
		blas_axpy(blas_hdl, kOne, &z, &p);
	}

	/* store solution for warm start in x0 (alias of z) */
	vector_memcpy_vv(&x0, x);
	if (!err)
		helper->never_solved = 0;

	if (h->norm_r > tol)
		if (!quiet){
			printf(nocvg_fmt, k, maxiter);
			printf(fmt, tol, h->norm_r, k);
		}

	*iters = k;
	return err;
}

uint pcg(operator * op, operator * pre_cond, vector * b, vector * x,
	const ok_float rho, const ok_float tol, const size_t maxiter,
	const int quiet)
{
	uint iters;
	ok_status err = OPTKIT_SUCCESS;
	pcg_helper * helper = pcg_helper_alloc(op->size1, op->size2);

	err = pcg_nonallocating(helper, op, pre_cond, b, x, rho, tol,
		maxiter, quiet, &iters);
	if (err)
		iters = INT_MAX;

	pcg_helper_free(helper);
	return iters;
}

void * pcg_init(size_t m, size_t n)
{
	return (void *) pcg_helper_alloc(m, n);
}

uint pcg_solve(void * pcg_work, operator * op, operator * pre_cond,
	vector * b, vector * x, const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet)
{
	uint iters;
	if ( pcg_nonallocating((pcg_helper *) pcg_work, op, pre_cond, b, x,
		rho, tol, maxiter, quiet, &iters) )
		return INT_MAX;
	else
		return iters;
}

ok_status pcg_finish(void * pcg_work)
{
	return pcg_helper_free((pcg_helper *) pcg_work);
}

#ifdef __cplusplus
}
#endif
