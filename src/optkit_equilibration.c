#include "optkit_equilibration.h"

#ifdef __cplusplus
extern "C" {
#endif

void equillib_version(int * maj, int * min, int * change, int * status)
{
	* maj = OPTKIT_VERSION_MAJOR;
	* min = OPTKIT_VERSION_MINOR;
	* change = OPTKIT_VERSION_CHANGE;
	* status = (int) OPTKIT_VERSION_STATUS;
}

void sinkhorn_knopp(void * linalg_handle, ok_float * A_in, matrix * A_out,
	vector * d, vector * e, enum CBLAS_ORDER ord)
{
	size_t m = A_out->size1, n = A_out->size2, k, NUMITER=10;
	ok_float sqrtm, sqrtn, nrm_d2, nrm_e2, fac;
	vector a = (vector) {0, 0, OK_NULL};
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
	const ok_float kSinkhornConst = 1e-4;
	const ok_float kEps = 1e-2;
	const size_t kMaxIter = 300;
	ok_float norm_d, norm_e;
	size_t i;

	vector a = (vector) {0, 0, OK_NULL};
	vector d_diff = (vector) {0, 0, OK_NULL};
	vector e_diff = (vector) {0, 0, OK_NULL};
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
		vector_add_constant(e, kSinkhornConst / e->size);
		vector_recip(e);
		vector_scale(e, d->size);

		blas_gemv(linalg_handle, CblasNoTrans, kOne, A_out, e, kZero,
			d);
		vector_add_constant(d, kSinkhornConst / d->size);
		vector_recip(d);
		vector_scale(d, e->size);

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

void dense_l2(void * linalg_handle, ok_float * A_in, matrix * A_out,
	vector * d, vector * e, enum CBLAS_ORDER ord)
{
	size_t k, m = A_out->size1, n = A_out->size2;
	vector a = (vector){0, 0, OK_NULL};


	if (A_in == A_out->data) {
		printf("Error: Dense l2 equilibration requires \
			distinct arrays A_in and A_out.");
		return;
	}

	matrix_memcpy_ma(A_out, A_in, ord);

	if (n > m) {
		/* fat matrix routine */
		vector_set_all(e, kOne);
		for (k = 0; k < m; ++k) {
			matrix_row(&a, A_out, k);
			blas_dot_inplace(linalg_handle, &a, &a, d->data + k);
		}
		vector_sqrt(d);
		vector_recip(d);

		for (k = 0; k < n; ++k) {
			matrix_column(&a, A_out, k);
			vector_mul(&a, d);
		}
	} else {
		/* skinny matrix routine */
		vector_set_all(d, kOne);
		for (k = 0; k < n; ++k) {
			matrix_column(&a, A_out, k);
			blas_dot_inplace(linalg_handle, &a, &a, e->data + k);
		}
		vector_sqrt(e);
		vector_recip(e);

		for (k = 0; k < m; ++k) {
			matrix_row(&a, A_out, k);
			vector_mul(&a, e);
		}
	}
}

#ifdef __cplusplus
}
#endif