#include "optkit_equilibration.h"

#ifdef __cplusplus
extern "C" {
#endif

void 
equillib_version(int * maj, int * min, int * change, int * status){
    * maj = OPTKIT_VERSION_MAJOR;
    * min = OPTKIT_VERSION_MINOR;
    * change = OPTKIT_VERSION_CHANGE;
    * status = (int) OPTKIT_VERSION_STATUS;
}

void 
sinkhorn_knopp(void * linalg_handle, ok_float * A_in, matrix * A_out, 
	vector * d, vector * e, CBLAS_ORDER_t ord){

	size_t m = A_out->size1, n = A_out->size2, k, NUMITER=10;
	ok_float sqrtm, sqrtn, nrm_d2, nrm_e2, fac;
	vector a = (vector) {0, 0, OK_NULL};
	sqrtm = MATH(sqrt)( (ok_float) m);
	sqrtn = MATH(sqrt)( (ok_float) n);

	if (A_in == A_out->data){
		printf("Error: Sinkhorn-Knopp equilibration requires \
			distinct arrays A_in and A_out.");
		return;
	}
	matrix_memcpy_ma(A_out, A_in, ord);
	matrix_abs(A_out);
	vector_set_all(d, kOne);

	/* repeat NUMITER times */
	for (k = 0; k < NUMITER; ++k){
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
void 
dense_l2(void * linalg_handle, ok_float * A_in, matrix * A_out, 
	vector * d, vector * e, CBLAS_ORDER_t ord){

	size_t k, m = A_out->size1, n = A_out->size2;
	vector a = (vector){0, 0, OK_NULL};

	if (A_in != A_out->data)
		matrix_memcpy_ma(A_out, A_in, ord);

	if (n > m){ 
		/* fat matrix routine */
		vector_set_all(e, kOne);
		for (k = 0; k < m; ++k){
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
		for (k = 0; k < n; ++k){
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

