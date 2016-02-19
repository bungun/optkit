#ifndef OPTKIT_CG_H_
#define OPTKIT_CG_H_

#include "optkit_abstract_operator.h"
#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CGLS helper struct */
typedef struct cgls_helper{
	vector p, q, r, s;
	ok_float norm_s, norm_s0, norm_x, xmax;
	ok_float alpha, beta, delta, gamma, gamma_prev, shrink;
	const ok_float kEps;
	void * blas_handle;
	int blas_handle_provided;
} cg_helper;


cgls_helper * cgls_helper_alloc(size_t m, size_t n,
	void * blas_handle);
void cgls_helper_free(cgls_helper * helper);

/* PCG helper struct */
typedef struct pcg_helper{
	vector p, q, r, z, temp;
	ok_float alpha, gamma, gamma_prev;
	void * blas_handle;
	int blas_handle_provided;
	int never_solved;
} cg_helper;


pcg_helper * pcg_helper_alloc(size_t m, size_t n,
	void * blas_handle);
void pcg_helper_free(pcg_helper * helper);


/* CGLS calls */
/* core method */
uint cgls_nonallocating(cgls_helper * helper,
	operator * op, vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet);

/* convenience wrappers */
uint cgls(operator * op, vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet);

void * cgls_easy_init(size_t m, size_t n);
uint cgls_easy_solve(void * cgls_work,
	operator * op, vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet);
void cgls_easy_finish(void * cgls_work);

/* Preconditioned CG calls */
/* core method */
uint pcg_nonallocating(pcg_helper * helper,
	operator * op, operator * pre_cond,
	vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet);

/* convenience wrappers */
uint pcg(operator * op, operator * pre_cond,
	vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet);

void * pcg_easy_init(size_t m, size_t n);
uint pcg_easy_solve(void * pcg_work,
	operator * op, operator * pre_cond,
	vector * b, vector * x,
	const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet);
void pcg_easy_finish(void * pcg_work);


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CG_H_ */