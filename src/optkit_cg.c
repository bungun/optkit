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
	cgls_helper * h;
	h = malloc(sizeof(*h));
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
	vector_calloc(&(h->p), n);
	vector_calloc(&(h->q), m);
	vector_calloc(&(h->r), m);
	vector_calloc(&(h->s), n);
	blas_make_handle(&(h->blas_handle));
	return h;
}

void cgls_helper_free(cgls_helper * helper)
{
	blas_destroy_handle(helper->blas_handle);
	vector_free(&(helper->p));
	vector_free(&(helper->q));
	vector_free(&(helper->r));
	vector_free(&(helper->s));
	ok_free(helper);
}

pcg_helper * pcg_helper_alloc(size_t m, size_t n)
{
	pcg_helper * h;
	h = malloc(sizeof(*h));
	h->norm_r = (ok_float) 0;
	h->alpha = (ok_float) 0;
	h->gamma = (ok_float) 0;
	h->gamma_prev = (ok_float) 0;
	vector_calloc(&(h->p), n);
	vector_calloc(&(h->q), n);
	vector_calloc(&(h->r), n);
	vector_calloc(&(h->z), n);
	vector_calloc(&(h->temp), m);
	blas_make_handle(&(h->blas_handle));
	h->never_solved = 1;
	return h;
}

void pcg_helper_free(pcg_helper * helper)
{
	blas_destroy_handle(helper->blas_handle);
	vector_free(&(helper->p));
	vector_free(&(helper->q));
	vector_free(&(helper->r));
	vector_free(&(helper->z));
	vector_free(&(helper->temp));
	ok_free(helper);
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
uint cgls_nonallocating(cgls_helper * helper,
		operator * op, vector * b, vector * x,
		const ok_float rho, const ok_float tol,
		const size_t maxiter, const int quiet)
{
	/* convenience abbreviations */
	cgls_helper * h = helper;
	vector p = h->p;
	vector q = h->q;
	vector r = h->r;
	vector s = h->s;
	void * blas_hdl = h->blas_handle;

	/* variable and constant declarations */
	char fmt[] = "%5d %9.2e %12.5e\n";
	uint k, flag = 0;
	int indefinite = 0, converged = 0;
	const ok_float kNegRho = -rho;

	/* initialization */
	h->norm_x = blas_nrm2(blas_hdl, x);

	/* r = b - Ax */
	vector_memcpy_vv(&r, b);
	if (h->norm_x > 0) {
		op->fused_apply(op->data, -kOne, x, kOne, &r);
	}

	/* s = A'*r - rho * x */
	op->adjoint(op->data, &r, &s);
	if (h->norm_x > 0)
		blas_axpy(blas_hdl, kNegRho, x, &s);

	/* p = s */
	vector_memcpy_vv(&p, &s);

	/* initial norms (norm_x calculated above) */
	h->norm_s0 = h->norm_s = blas_nrm2(blas_hdl, &s);
	h->gamma = h->norm_s0 * h->norm_s0;
	h->xmax = h->norm_x;

	if (h->norm_s < kEps)
		flag = 1;

	if (!quiet && !flag)
		printf("    k    norm x        resNE\n");

	/* ------------------- CGLS ----------------- */
	for (k = 0; k < maxiter && !flag; ++k) {
		/* q = Ap */
		op->apply(op->data, &p, &q);

		/*
		 * NOTE: CHRIS' COMMENT SAYS
		 * delta = ||p||_2^2 + rho * ||q||_2^2
		 * BUT THE CODE PERFORMS:
		 * delta = ||q||_2^2 + rho * ||p||_2^2
		 */
		h->delta = blas_dot(blas_hdl, &q, &q) +
				rho * blas_dot(blas_hdl, &p, &p);

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
		h->norm_s = blas_nrm2(blas_hdl, &s);
		h->gamma_prev = h->gamma;
		h->gamma = h->norm_s * h->norm_s;
		h->beta = h->gamma / h->gamma_prev;

		/* p = s + beta * p */
		vector_scale(&p, h->beta);
		blas_axpy(blas_hdl, kOne, &s, &p);

		/* convergence check */
		h->norm_x = blas_nrm2(blas_hdl, x);
		h->xmax = (h->norm_x > h->xmax) ? h->norm_x : h->xmax;
		converged = (h->norm_s < h->norm_s0 * tol) ||
			(h->norm_x * tol > 1);
		if (!quiet && (converged || (k + 1) % 10 == 0))
			printf(fmt, k + 1, h->norm_x, h->norm_s / h->norm_s0);
		if (converged)
			break;
	}

	/* determine exit status */
	h->shrink = h->norm_x / h->xmax;
	if (k == maxiter)
		flag = 2;
	else if (indefinite)
		flag = 3;
	else if (h->shrink * h->shrink <= tol)
		flag = 4;

	return flag;
}

uint cgls(operator * op, vector * b, vector * x, const ok_float rho,
	const ok_float tol, const size_t maxiter, const int quiet)
{
	uint flag;

	cgls_helper * helper = cgls_helper_alloc(op->size1, op->size2);

	/* CGLS call */
	flag = cgls_nonallocating(helper, op, b, x, rho, tol, maxiter, quiet);

	cgls_helper_free(helper);

	return flag;
}

void * cgls_easy_init(size_t m, size_t n)
{
	return (void *) cgls_helper_alloc(m, n);
}

uint cgls_easy_solve(void * cgls_work, operator * op, vector * b, vector * x,
	const ok_float rho, const ok_float tol, const size_t maxiter, int quiet)
{
	return cgls_nonallocating( (cgls_helper *) cgls_work, op, b, x, rho,
		tol, maxiter, quiet);
}

void cgls_easy_finish(void * cgls_work)
{
	cgls_helper_free((cgls_helper *) cgls_work);
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
void diagonal_preconditioner(operator * op, vector * p, ok_float rho)
{
	vector ej = (vector) {0, 0, OK_NULL};
	vector ej_sub = (vector) {0, 0, OK_NULL};
	vector a = (vector) {0, 0, OK_NULL};
	vector iaa = (vector) {0, 0, OK_NULL};
	vector p_sub = (vector) {0, 0, OK_NULL};

	vector_calloc(&ej, op->size2);
	vector_calloc(&a, op->size2);
	vector_calloc(&iaa, op->size2);

	ok_float col_norm_sq;
	size_t i;

	void * blas_handle;
	blas_make_handle(&blas_handle);

	vector_scale(p, 0);

	printf("%s %f\n", "RHO C", rho);
	for (i = 0; i < op->size2; ++i) {
		vector_scale(&ej, kZero);
		vector_subvector(&ej_sub, &ej, i, 1);
		vector_add_constant(&ej_sub, 1);
		op->apply(op->data, &ej, &a);
		op->adjoint(op->data, &a, &iaa);

		col_norm_sq = blas_dot(blas_handle, &iaa, &iaa);
		vector_subvector(&p_sub, p, i, 1);
		vector_add_constant(&p_sub, col_norm_sq + rho);
	}

	vector_recip(p);

	vector_free(&ej);
	vector_free(&a);
	vector_free(&iaa);

	blas_destroy_handle(blas_handle);
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
uint pcg_nonallocating(pcg_helper * helper, operator * op, operator * pre_cond,
	vector * b, vector * x, const ok_float rho, const ok_float tol,
	const size_t maxiter, const int quiet)
{
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
	h->norm_r = MATH(sqrt)(blas_dot(blas_hdl, &r, &r));
	if (blas_nrm2(blas_hdl, &r) < kNormTol)
	        return 0;

	/* p = Mr (apply preconditioner) */
	pre_cond->apply(pre_cond->data, &r, &p);
	/* gamma = r'Mr */
	h->gamma = blas_dot(blas_hdl, &r, &p);

	for (k = 0; k < maxiter; ++k) {
		/* q = (rho * I + A'A)p */
		op->apply(op->data, &p, &temp);
		op->adjoint(op->data, &temp, &q);
		blas_axpy(blas_hdl, rho, &p, &q);

		/* alpha = <p, r> / <p, q> */
		h->alpha = h->gamma / blas_dot(blas_hdl, &p, &q);

		/* x += alpha * p */
		/* r -= alpha * q */
		blas_axpy(blas_hdl, +h->alpha, &p, x);
		blas_axpy(blas_hdl, -h->alpha, &q, &r);

		/* check convergence */
		h->norm_r = MATH(sqrt)(blas_dot(blas_hdl, &r, &r));
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
		h->gamma = blas_dot(blas_hdl, &r, &z);

		/* p = p * gamma / gamma_prev + Mr */
		vector_scale(&p, h->gamma / h->gamma_prev);
		blas_axpy(blas_hdl, kOne, &z, &p);
	}

	/* store solution for warm start in x0 (alias of z) */
	vector_memcpy_vv(&x0, x);
	helper->never_solved = 0;

	if (h->norm_r > tol)
		if (!quiet){
			printf(nocvg_fmt, k, maxiter);
			printf(fmt, tol, h->norm_r, k);
		}

	return k;
}

uint pcg(operator * op, operator * pre_cond, vector * b, vector * x,
	const ok_float rho, const ok_float tol, const size_t maxiter,
	const int quiet)
{
	uint flag;
	pcg_helper * helper = pcg_helper_alloc(op->size1, op->size2);

	flag = pcg_nonallocating(helper, op, pre_cond, b, x, rho, tol,
		maxiter, quiet);

	pcg_helper_free(helper);
	return flag;
}

void * pcg_easy_init(size_t m, size_t n)
{
	return (void *) pcg_helper_alloc(m, n);
}

uint pcg_easy_solve(void * pcg_work, operator * op, operator * pre_cond,
	vector * b, vector * x, const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet)
{
	return pcg_nonallocating( (pcg_helper *) pcg_work, op, pre_cond, b, x,
		rho, tol, maxiter, quiet);
}

void pcg_easy_finish(void * pcg_work)
{
	pcg_helper_free((pcg_helper *) pcg_work);
}

#ifdef __cplusplus
}
#endif