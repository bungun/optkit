#include "optkit_rand.h"
#include "optkit_equilibration.h"

#ifdef __cplusplus
extern "C" {
#endif

void sinkhorn_knopp(void * linalg_handle, ok_float * A_in, matrix * A_out,
	vector * d, vector * e, enum CBLAS_ORDER ord)
{
	size_t m = A_out->size1, n = A_out->size2, k, NUMITER=10;
	ok_float sqrtm, sqrtn, nrm_d2, nrm_e2, fac;
	vector a;
	a.data = OK_NULL;

	sqrtm = MATH(sqrt)( (ok_float) m);
	sqrtn = MATH(sqrt)( (ok_float) n);


	if (A_in == A_out->data) {
		printf("Error: Sinkhorn-Knopp equilibration requires \
			distinct arrays A_in and A_out.");
		return;
	}

	matrix_memcpy_ma(A_out, A_in, ord);
	matrix_abs(A_out);
	vector_set_all(d, kOne);

	/* repeat NUMITER times */
	for (k = 0; k < NUMITER; ++k) {
		blas_gemv(linalg_handle, CblasTrans, kOne, A_out, d, kZero, e);
		vector_recip(e);
		blas_gemv(linalg_handle, CblasNoTrans, kOne, A_out, e, kZero, d);
		vector_recip(d);
		nrm_d2 = blas_dot(linalg_handle, d, d);
		nrm_e2 = blas_dot(linalg_handle, e, e);
		fac = MATH(sqrt)(MATH(sqrt)(nrm_e2 / nrm_d2) * sqrtm / sqrtn);
		vector_scale(d, fac);
		vector_scale(e, (ok_float) 1/fac);
	}

	matrix_memcpy_ma(A_out, A_in, ord);
	for (k = 0; k < m; ++k) {
		matrix_row(&a, A_out, k);
		vector_mul(&a, e);
	}
	for (k = 0; k < n; ++k) {
		matrix_column(&a, A_out, k);
		vector_mul(&a, d);
	}
}


void regularized_sinkhorn_knopp(void * linalg_handle, ok_float * A_in,
	matrix * A_out, vector * d, vector * e, enum CBLAS_ORDER ord)
{
	const ok_float kSinkhornConst = (ok_float) 1e-4;
	const ok_float kEps = (ok_float) 1e-2;
	const size_t kMaxIter = 300;
	ok_float norm_d, norm_e;
	size_t i;

	vector a, d_diff, e_diff;
	a.data = OK_NULL;
	d_diff.data = OK_NULL;
	e_diff.data = OK_NULL;

	vector_calloc(&d_diff, A_out->size1);
	vector_calloc(&e_diff, A_out->size2);

	norm_d = norm_e = 1;

	matrix_memcpy_ma(A_out, A_in, ord);
	matrix_abs(A_out);
	vector_set_all(d, kOne);
	vector_scale(e, kZero);

	/* optional argument ok_float pnorm? */
	/*
	if (pnorm != 1) {
		matrix_pow(A, pnorm)
	}
	*/

	for (i = 0; i < kMaxIter; ++i){
		blas_gemv(linalg_handle, CblasTrans, kOne, A_out, d, kZero, e);
		vector_add_constant(e, kSinkhornConst / (ok_float) e->size);
		vector_recip(e);
		vector_scale(e, (ok_float) d->size);

		blas_gemv(linalg_handle, CblasNoTrans, kOne, A_out, e, kZero,
			d);
		vector_add_constant(d, kSinkhornConst / (ok_float) d->size);
		vector_recip(d);
		vector_scale(d, (ok_float) e->size);

		blas_axpy(linalg_handle, -kOne, d, &d_diff);
		blas_axpy(linalg_handle, -kOne, e, &e_diff);

		norm_d = MATH(sqrt)(blas_dot(linalg_handle, &d_diff, &d_diff));
		norm_e = MATH(sqrt)(blas_dot(linalg_handle, &e_diff, &e_diff));

		if ((norm_d < kEps) && (norm_e < kEps))
			break;

		vector_memcpy_vv(&d_diff, d);
		vector_memcpy_vv(&e_diff, e);
	}

	/* optional argument ok_float pnorm? */
	/*
	if (pnorm != 1) {
		vector_pow(d, kOne / pnorm)
		vector_pow(e, kOne / pnorm)
	}
	*/

	matrix_memcpy_ma(A_out, A_in, ord);
	for (i = 0; i < A_out->size1; ++i) {
		matrix_row(&a, A_out, i);
		vector_mul(&a, e);
	}
	for (i = 0; i < A_out->size2; ++i) {
		matrix_column(&a, A_out, i);
		vector_mul(&a, d);
	}

	vector_free(&d_diff);
	vector_free(&e_diff);
}

// void dense_l2(void * linalg_handle, ok_float * A_in, matrix * A_out,
// 	vector * d, vector * e, enum CBLAS_ORDER ord)
// {
// 	size_t k, m = A_out->size1, n = A_out->size2;
// 	vector a;
// 	a.data = OK_NULL;

// 	if (A_in == A_out->data) {
// 		printf("Error: Dense l2 equilibration requires \
// 			distinct arrays A_in and A_out.");
// 		return;
// 	}

// 	matrix_memcpy_ma(A_out, A_in, ord);

// 	if (n > m) {
// 		/* fat matrix routine */
// 		vector_set_all(e, kOne);
// 		for (k = 0; k < m; ++k) {
// 			matrix_row(&a, A_out, k);
// 			blas_dot_inplace(linalg_handle, &a, &a, d->data + k);
// 		}
// 		vector_sqrt(d);
// 		vector_recip(d);

// 		for (k = 0; k < n; ++k) {
// 			matrix_column(&a, A_out, k);
// 			vector_mul(&a, d);
// 		}
// 	} else {
// 		/* skinny matrix routine */
// 		vector_set_all(d, kOne);
// 		for (k = 0; k < n; ++k) {
// 			matrix_column(&a, A_out, k);
// 			blas_dot_inplace(linalg_handle, &a, &a, e->data + k);
// 		}
// 		vector_sqrt(e);
// 		vector_recip(e);

// 		for (k = 0; k < m; ++k) {
// 			matrix_row(&a, A_out, k);
// 			vector_mul(&a, e);
// 		}
// 	}
// }

#ifndef OPTKIT_NO_OPERATOR_EQUIL
ok_status operator_regularized_sinkhorn(void * linalg_handle, operator * A,
	vector * d, vector * e, const ok_float pnorm)
{
	transformable_operator * transform = OK_NULL;
	void * A_temp = OK_NULL;
	const ok_float kSinkhornConst = (ok_float) 1e-4;
	const ok_float kEps = (ok_float) 1e-2;
	const size_t kMaxIter = 300;
	ok_float norm_d, norm_e;
	ok_status err = OPTKIT_SUCCESS;
	size_t k;
	int blas_handle_provided = 1;
	vector d_diff, e_diff;
	d_diff.data = OK_NULL;
	e_diff.data = OK_NULL;

	if (!linalg_handle) {
		blas_make_handle(&linalg_handle);
		blas_handle_provided = 0;
	}

	if (A->kind == OkOperatorDense) {
		transform = dense_operator_to_transformable(A);
	} else if (A->kind == OkOperatorSparseCSC ||
		   A->kind == OkOperatorSparseCSR) {
		transform = sparse_operator_to_transformable(A);
	} else {
		printf("\n%s", "ERROR: operator_regularized_sinkhorn only ");
		printf("%s\n", "defined for dense and sparse operators");
		return OPTKIT_ERROR;
	}

	vector_calloc(&d_diff, A->size1);
	vector_calloc(&e_diff, A->size2);

	norm_d = norm_e = 1;

	A_temp = transform->export(A);
	err = (A_temp == OK_NULL);

	if (!err)
		err = transform->abs(A);

	vector_set_all(d, kOne);

	if (pnorm != 1 && !err)
		err = transform->pow(A, pnorm);

	for (k = 0; k < kMaxIter && !err; ++k) {
		A->adjoint(A->data, d, e);
		vector_add_constant(e, kSinkhornConst / (ok_float) e->size);
		vector_recip(e);
		vector_scale(e, (ok_float) d->size);
		A->apply(A->data, e, d);
		vector_add_constant(d, kSinkhornConst / (ok_float) d->size);
		vector_recip(d);
		vector_scale(d, (ok_float) e->size);

		norm_d = MATH(sqrt)(blas_dot(linalg_handle, &d_diff, &d_diff));
		norm_e = MATH(sqrt)(blas_dot(linalg_handle, &e_diff, &e_diff));

		if ((norm_d < kEps) && (norm_e < kEps))
			break;

		vector_memcpy_vv(&d_diff, d);
		vector_memcpy_vv(&e_diff, e);
	}

	if (pnorm != 1 && !err) {
		vector_pow(d, kOne / pnorm);
		vector_pow(e, kOne / pnorm);
	}

	if (!err) {
		A_temp = transform->import(A, A_temp);
	}

	if (!err) {
		err = transform->scale_left(A, d);
		err = transform->scale_right(A, e);
	}

	if (A_temp)
		ok_free(A_temp);

	if (transform)
		ok_free(transform);

	vector_free(&d_diff);
	vector_free(&e_diff);

	if (!blas_handle_provided)
		blas_destroy_handle(linalg_handle);

	return err;
}

/* STUB */
ok_status operator_equilibrate(void * linalg_handle, operator * A,
	vector * d, vector * e, const ok_float pnorm)
{
	return OPTKIT_ERROR;
}

/*
 * given linear operator A, estimate the operator norm of A as
 *
 *	||A|| ~ ||(A'A)^n x|| / ||A(A'A)^(n-1)x||
 *
 * where x is a random vector and n <= kNormIter is a
 * given number of iterations.
 */
ok_float operator_estimate_norm(void * linalg_handle, operator * A)
{
	const ok_float kNormTol = (ok_float) 1e-5;
	const uint kNormIter = 50u;

	ok_float norm_est = kZero, norm_est_prev, norm_x;
	vector x, Ax;
	uint i;

	x.data = OK_NULL;
	Ax.data = OK_NULL;

	vector_calloc(&x, A->size2);
	vector_calloc(&Ax, A->size1);

	ok_rand(x.data, x.size);

	for (i = 0; i < kNormIter; ++i) {
		norm_est_prev = norm_est;
		A->apply(A->data, &x, &Ax);
		A->adjoint(A->data, &Ax, &x);
		norm_x = blas_nrm2(linalg_handle, &x);
		norm_est = norm_x / blas_nrm2(linalg_handle, &Ax);
		vector_scale(&x, 1 / norm_x);
		if (MATH(fabs)(norm_est_prev - norm_est) <= kNormTol * norm_est)
			break;
	}
	vector_free(&x);
	vector_free(&Ax);
	return norm_est;
}
#else
ok_status operator_regularized_sinkhorn(void * linalg_handle, operator * A,
	vector * d, vector * e, const ok_float pnorm)
{
	return OPTKIT_ERROR;
}

ok_status operator_equilibrate(void * linalg_handle, operator * A,
	vector * d, vector * e, const ok_float pnorm)
{
	return OPTKIT_ERROR;
}

ok_float operator_estimate_norm(void * linalg_handle, operator * A)
{
	return OPTKIT_ERROR;
}
#endif /* ndef OPTKIT_NO_OPERATOR_EQUIL */

#ifdef __cplusplus
}
#endif
