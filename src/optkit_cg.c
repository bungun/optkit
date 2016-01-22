/* The CGLS routine below is adapted from 	*/
/* Chris Fougner's software package POGS 	*/
/* (original license below)					*/

/* The preconditioned CG routine below is adapted 	*/
/* from Brendan O'Donoghue's software package SCS 	*/
/* (original license below)							*/


/* original CGLS license from cgls.h and cgls.cuh in POGS:
////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015, Christopher Fougner                                    //
// All rights reserved.                                                       //
//                                                                            //
// Redistribution and use in source and binary forms, with or without         //
// modification, are permitted provided that the following conditions are     //
// met:                                                                       //
//                                                                            //
//   1. Redistributions of source code must retain the above copyright        //
//      notice, this list of conditions and the following disclaimer.         //
//                                                                            //
//   2. Redistributions in binary form must reproduce the above copyright     //
//      notice, this list of conditions and the following disclaimer in the   //
//      documentation and/or other materials provided with the distribution.  //
//                                                                            //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        //
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  //
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR //
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          //
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      //
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        //
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         //
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     //
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       //
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         //
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               //
////////////////////////////////////////////////////////////////////////////////

*/

/* original license for SCS */
/*
The MIT License (MIT)

Copyright (c) 2012 Brendan O'Donoghue (bodonoghue85@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/



#include "optkit_cg.h"

#ifdef __cplusplus
extern "C" {
#endif


cgls_helper * cgls_helper_alloc(size_t m, size_t n, 
	void * blas_handle){

	cgls_helper * h = (cgls_helper *) malloc(
		sizeof(cgls_helper) );

	vector_calloc(&(h->p), n);
	vector_calloc(&(h->q), m);
	vector_calloc(&(h->r), m);
	vector_calloc(&(h->s), n);

	h->kEps = (ok_float) 1e-16;
	if (blas_hdl != OK_NULL){
		h->blas_handle = blas_handle;
		h->blas_handle_provided = 1;
	}
	else{
		blas_make_handle(&(h->blas_handle));
		h->blas_handle_provided = 0;
	}

	return h;
}
void cgls_helper_free(cgls_helper * helper){
	if (!helper->blas_handle_provided)
		blas_destroy_handle(h->blas_handle);
	vector_free(&(helper->p));
	vector_free(&(helper->q));
	vector_free(&(helper->r));
	vector_free(&(helper->s));
	ok_free(helper);
}


pcg_helper * pcg_helper_alloc(size_t m, size_t n, 
	void * blas_handle){

	cgls_helper * h = (pcg_helper *) malloc(
		sizeof(pcg_helper) );

	vector_calloc(&(h->p), n);
	vector_calloc(&(h->q), n);
	vector_calloc(&(h->r), n);
	vector_calloc(&(h->z), n);
	vector_calloc(&(h->temp), m);

	if (blas_hdl != OK_NULL){
		h->blas_handle = blas_handle;
		h->blas_handle_provided = 1;
	}
	else{
		blas_make_handle(&(h->blas_handle));
		h->blas_handle_provided = 0;
	}

	h->never_solved = 1;

	return h;
}
void pcg_helper_free(pcg_helper * helper){
	if (!helper->blas_handle_provided)
		blas_destroy_handle(h->blas_handle);
	vector_free(&(helper->p));
	vector_free(&(helper->q));
	vector_free(&(helper->r));
	vector_free(&(helper->z));
	vector_free(&(helper->temp));
	ok_free(helper);
}


/* ---------------------------------------------------------------- */
/*  CGLS Conjugate Gradient Least squares 							*/
/* 																	*/
/*  Attempts to solve the least squares problem 					*/
/* 																	*/
/*    min. ||Ax - b||_2^2 + rho ||x||_2^2 							*/
/* 																	*/
/*  using the Conjugate Gradient for Least Squares method. 			*/
/*  This is more stable than applying CG to the normal equations. 	*/
/* 																	*/
/* ref: Chris Fougner, POGS 									   	*/
/* https://github.com/foges/pogs/blob/master/src/cpu/include/cgls.h	*/
/* ---------------------------------------------------------------- */


uint cgls_nonallocating(cgls_helper * helper, operator_t * op,
	vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, const int quiet){

	/* convenience abbreviations */
	cgls_helper * h = helper;
	vector * p = &(h->p), * q = &(h->q);
	vector * r = &(h->r), * s = &(h->s);
	void * blas_hdl = h->blas_handle;

	/* variable and constant declarations */
	char fmt[] = "%5d %9.2e %12.5g\n";
	uint k, flag = 0;
	int indefinite = 0, converged = 0;
	const ok_float kNegRho = -rho;

	/* initialization */
	norm_x = blas_nrm2(blas_hdl, x);

	/* r = b - Ax */
	vector_memcpy_vv(r, b);
	if (normx > 0){
		op->fused_apply(op->data, -kOne, x, kOne, r);
		__DEVICE_SYNCHRONIZE__();
	}

	/* s = A'*r - rho * x */
	vector_memcpy_vv(s, x);
	if (h->norm_x > 0){
		op->fused_adjoint(op->data, kOne, r, kNegRho, s);
	}	

	/* p = s */
	vector_memcpy_vv(p, s);

	/* initial norms (norm_x calculated above) */
	h->norm_s0 = h->norm_s = blas_nrm2(blas_hdl, s);
	h->gamma = h->norm_s0 * h->norm_s0;
	h->xmax = h->norm_x;

	if (h->norm_s < h->kEps)
		flag = 1;

	if (!quiet && !flag)
		printf("    k     norm x       resNE\n");


	/* ------------------- CGLS ----------------- */
	for (k = 0; k < maxiter && !flag; ++k) {

		/* q = Ap */
		op->apply(op->data, q, p);

		/* NOTE: CHRIS' COMMENT SAYS:			*/
		/* delta = ||p||_2^2 + rho * ||q||_2^2 	*/
		/* BUT THE CODE PERFORMS: 				*/
		/* delta = ||q||_2^2 + rho * ||p||_2^2 	*/
		h->delta = blas_dot(blas_hdl, q, q) + 
			rho * blas_dot(blas_hdl, p, p);

		if (h->delta <= 0)
		  indefinite = 1;
		if (h->delta == 0)
		  h->delta = kEps;
		
		h->alpha = h->gamma / h->delta;
		/* x = x + alpha * p */
		/* r = r - alpha * q */
		blas_axpy(blas_hdl, h->alpha, p, x);
		blas_axpy(blas_hdl, -(h->alpha), q, r);

		/* s = A'r - rho * x */
		vector_memcpy_vv(s, x);
		op->fused_adjoint(op->data, kOne, r, kNegRho, x);

		/* compute beta */
		h->norm_s = blas_nrm2(blas_hdl, s);
		h->gamma_prev = h->gamma;
		gamma = h->norm_s * h->norm_s;
		beta = h->gamma / h->gamma_prev;

		/* p = s + beta * p */
		blas_axpy(blas_hdl, h->beta, p, s);
		vector_memcpy_vv(p, s);


		/* convergence check */
		h->norm_x = blas_nrm2(blas_hdl, x);
		h->xmax = (h->norm_x > h->xmax) ? h->norm_x : h->xmax;
		converged = (h->norm_s < h->norm_s0 * tol) || 
			(h->norm_x * tol > 1);
		if (!quiet && (converged || k % 10 == 0))
			printf(fmt, k, h->norm_x, h->norm_s / h->norm_s0);
		if (converged)
			break;
	}

	/* determine exit status */
	h->shrink = h->norm_x / h->xmax;
	if (k == maxit)
		flag = 2;
	else if (indefinite)
		flag = 3;
	else if (h->shrink * h->shrink <= tol)
		flag = 4;
	/* ------------------------------------------ */
	h->never_solved = 0;

	return flag;

}

uint cgls(void * blas_handle, operator_t * op, 
	vector * input, vector * output,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, const int quiet){

	int flag;

	/* memory allocation */
	cgls_helper * helper = cgls_helper_alloc(
		op->size1, op->size2, blas_handle);

	/* CGLS call */
	flag = cgls_nonallocating(helper, op, input, output, 
		rho, tol, maxiter, quiet);

	/* memory cleanup */
	cgls_helper_free(helper);

	return flag;
}

void * cgls_easy_init(size_t m, size_t n){
	return (void *) cgls_helper_alloc(m, n, OK_NULL);
}
uint cgls_easy_solve(void * cgls_work,
	operator_t * op, vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet){

	return cgls_nonallocating(
		(cgls_helper *) helper, op, input, output, 
		rho, tol, maxiter, quiet);
}

void cgls_easy_finish(void * cgls_work){
	cgls_helper_free((cgls_helper *) cgls_work);
}

/* --------------------------------------------------------------------	*/
/* Preconditioned Conjugate Gradient (solve)							*/
/*  																	*/
/* Attempts to solve 													*/
/*  																	*/
/* 		M(rho * I + A'A)x = b											*/
/* 																		*/
/* to specified tolerance within maxiter CG iterations					*/
/*  																	*/
/* Solution written to method parameter x, and stored in helper->z 		*/
/* Warm starts from helper->z if helper->never_ran is false 			*/																	*/
/*  																	*/
/* ref: Brendan O'Donoghue, SCS 									   	*/
/* https://github.com/cvxgrp/scs/blob/master/linsys/indirect/private.c 	*/
/* -------------------------------------------------------------------- */

uint pcg_nonallocating(pcg_helper * helper, 
	operator_t * op, operator_t * pre_cond, 
	vector * b, vector * x, const ok_float rho,
	const ok_float tol, const size_t maxiter, const int quiet){

	/* convenience abbreviations */
	cgls_helper * h = helper;
	vector * p = &(h->p), * q = &(h->q);  	/* iterate vectors */
	vector * r = &(h->r), * z = &(h->z);   	/* residual, preconditioned var */
	vector * x0 = &(h->z) 					/* alias for warmstart */
	vector * temp = &(h->temp); 			/* temp for A'A intermediate */ 

	void * blas_hdl = h->blas_handle;

	/* variable/constant declarations */
	uint k;
	const ok_float kNormTol = (tol < 1e-18) ? tol : (ok_float) 1e-18;
	char fmt [] = "tol: %.4e, resid: %.4e, iters: %u\n";


	/* initialization */
	if (h->never_solved){
		/* r = b */
		/* x = 0 */
		vector_memcpy_vv(r, b);
		vector_scale(x, kZero);
	} else {
		/* r = b - (rho * I + A'A)x0 */
		/* x = x0 */
		vector_memcpy_vv(r, b);
		op->apply(op->data, x0, temp);
		op->fused_apply(op->data, -kOne, temp, kOne, r);
		blas_axpy(blas_hdl, -rho, x0, r);
	}

    /* check to see if we need to run CG at all */
    if (blas_nrm2(blas_handle, r) < kNormTol)
        return 0;
   
	/* p = Mr (apply preconditioner) */
	pre_cond->apply(pre_cond->data, r, p);
	/* gamma = r'Mr */
	h->gamma = blas_dot(blas_hdl, r, p);

    for (k = 0; k < maxiter; ++j) {

    	/* q = (rho * I + A'A)p */
    	op->apply(op->data, p, temp);
    	op->fused_apply(op->data, kOne, temp, kZero, Tp);
    	blas_axpy(blas_hdl, rho, p, q);

    	/* alpha = <p, r> / <p, Tp> */
        h->alpha = h->gamma / blas_dot(blas_hdl, p, Tp);

        /* x = x + alpha * p */
        /* r = r - alpha * Tp */
        blas_axpy(blas_hdl, alpha, p, x);
        blas_axpy(blas_hdl, -alpha, Tp, r);

        /* check convergence */
		if (calcNorm(r, n) < tol) {
			if (!quiet)
				printf(fmt, tol, calcNorm(r, n), k + 1);
			k += 1;
			break;
		}
		h->gamma_prev = h->gamma;

		/* z = Mr */
		pre_cond->apply(pre_cond->data, r, z);
		/* gamma = r'Mr */
		h->gamma = blas_dot(blas_hdl, r, z);

		/* p = p * gamma / gamma_prev + Mr */
		vector_scale(p, h->gamma / h->gamma_prev);
		blas_axpy(blas_hdl, kOne, z, p);
	}

	/* store solution for warm start in x0 (alias of z) */
	vector_memcpy_vv(x0, x);
	helper->never_solved = 0;

	return k;
}

uint pcg(void * blas_handle, operator_t * op,
	operator_t * pre_cond,  
	vector * input, vector * output,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, const int quiet){

	int flag;

	/* memory allocation */
	pcg_helper * helper = pcg_helper_alloc(
		op->size1, op->size2, blas_handle);

	/* PCG call */
	flag = pcg_nonallocating(helper, op, pre_cond,
		input, output, rho, tol, maxiter, quiet);

	/* memory cleanup */
	pcg_helper_free(helper);

	return flag;
}

void * pcg_easy_init(size_t m, size_t n){
	return (void *) pcg_helper_alloc(m, n, OK_NULL);
}
uint pcg_easy_solve(void * pcg_work,
	operator_t * op, operator_t * pre_cond, 
	vector * b, vector * x, 
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet){

	return pcg_nonallocating(
		(pcg_helper *) pcg_work, op, pre_cond,
		input, output, rho, tol, maxiter, quiet);
}

void pcg_easy_finish(void * pcg_work){
	pcg_helper_free((pcg_helper *) pcg_work);
}


#ifdef __cplusplus
}
#endif