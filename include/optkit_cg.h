#ifndef OPTKIT_CG_H_
#define OPTKIT_CG_H_

#include "optkit_vector.h"
#include "optkit_abstract_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CGLS helper struct */
typedef struct cgls_helper{
	vector p, q, r, s;
	ok_float norm_s, norm_s0, norm_x, xmax;
	ok_float alpha, beta, delta, gamma, gamma_prev, shrink;
	void *blas_handle;
} cgls_helper;

cgls_helper * cgls_helper_alloc(size_t m, size_t n);
ok_status cgls_helper_free(cgls_helper *helper);

/* PCG helper struct */
typedef struct pcg_helper{
	vector p, q, r, z, temp;
	ok_float norm_r;
	ok_float alpha, gamma, gamma_prev;
	void *blas_handle;
	int never_solved;
} pcg_helper;

pcg_helper * pcg_helper_alloc(size_t m, size_t n);
ok_status pcg_helper_free(pcg_helper *helper);

/* CGLS calls */
/* core method */
ok_status cgls_nonallocating(cgls_helper *helper, abstract_operator *op,
	vector *b, vector *x, const ok_float rho, const ok_float tol,
	const size_t maxiter, int quiet, uint *flag);

/* convenience wrappers */
ok_status cgls(abstract_operator *op, vector *b, vector *x, const ok_float rho,
	const ok_float tol, const size_t maxiter, int quiet, uint *flag);

void * cgls_init(size_t m, size_t n);
ok_status cgls_solve(void *cgls_work, abstract_operator *op, vector *b,
	vector *x, const ok_float rho, const ok_float tol, const size_t maxiter,
	int quiet, uint *flag);
ok_status cgls_finish(void *cgls_work);

/* Preconditioned CG calls */
ok_status diagonal_preconditioner(abstract_operator *op, vector *p,
	ok_float rho);

/* core method */
ok_status pcg_nonallocating(pcg_helper *helper, abstract_operator *op,
	abstract_operator *pre_cond, vector *b, vector *x, const ok_float rho,
	const ok_float tol, const size_t maxiter, int quiet, uint *iters);

/* convenience wrappers */
ok_status pcg(abstract_operator *op, abstract_operator *pre_cond, vector *b,
	vector *x, const ok_float rho, const ok_float tol, const size_t maxiter,
	int quiet, uint *iters);

void * pcg_init(size_t m, size_t n);
ok_status pcg_solve(void *pcg_work, abstract_operator *op,
	abstract_operator *pre_cond, vector *b, vector *x, const ok_float rho,
	const ok_float tol, const size_t maxiter, int quiet, uint *iters);
ok_status pcg_finish(void *pcg_work);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CG_H_ */
