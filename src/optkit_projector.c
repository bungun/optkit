#include "optkit_projector.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Direct Projector methods */
void 
direct_projector_alloc(direct_projector * P, matrix * A){
	size_t mindim = (A->size1 >= A->size2) ? A->size1 : A->size2;

	P->A = A;
	P->L = (matrix *) malloc( sizeof(matrix) );
	matrix_calloc(P->L, mindim, mindim, A->rowmajor);
	P->skinny = (uint) mindim == A->size1;
	P->normalized = 0;
}

void 
direct_projector_free(direct_projector * P){ 
	matrix_free(P->L); 
	P->A = OK_NULL;
	ok_free(P);
}

void 
direct_projector_initialize(void * linalg_handle, direct_projector * P, 
	int normalize){
	
	vector diag = (vector){0, 0, OK_NULL};
	ok_float mean_diag = kZero;

	if (P->skinny)
		blas_gemm(linalg_handle, CblasTrans, CblasNoTrans, 
			kOne, P->A, P->A, kZero, P->L);
	else
		blas_gemm(linalg_handle, CblasNoTrans, CblasTrans, 
			kOne, P->A, P->A, kZero, P->L);


	matrix_diagonal(&diag, P->L);
	mean_diag = blas_asum(linalg_handle, &diag) / (ok_float) P->L->size1;
	P->normA =  MATH(sqrt)(mean_diag);

	if (normalize){
		matrix_scale(P->L, kOne / mean_diag);
		matrix_scale(P->A, kOne / P->normA);
	}

	P->normalized = normalize;
}

void 
direct_projector_project(void * linalg_handle, direct_projector * P, 
	vector * x_in, vector * y_in, vector * x_out, vector * y_out){

	if (P->skinny){
		vector_memcpy_vv(x_out, x_in);
		blas_gemv(linalg_handle, CblasTrans, kOne, P->A, y_in, kOne, x_out);
		linalg_cholesky_svx(linalg_handle, P->L, x_out);
		blas_gemv(linalg_handle, CblasNoTrans, kOne, P->A, x_out, kZero, y_out);

	} else {
		vector_memcpy_vv(y_out, y_in);
		blas_gemv(linalg_handle, CblasNoTrans, kOne, P->A, x_in, -kOne, y_out);
		linalg_cholesky_svx(linalg_handle, P->L, y_out);
		blas_gemv(linalg_handle, CblasTrans, -kOne, P->A, y_out, kZero, x_out);
		blas_axpy(linalg_handle, kOne, y_in, y_out);
		blas_axpy(linalg_handle, kOne, x_in, x_out);
	}
}


/* Indirect Projector methods */
void indirect_projector_alloc(direct_projector * P, matrix * A){
	P->A = A;
}
void indirect_projector_initialize(void * linalg_handle, direct_projector * P,
	int normalize){
	return;
}
void indirect_projector_project(void * linalg_handle, direct_projector * P, 
	vector * x_in, vector * y_in, vector * x_out, vector * y_out){
	return;
}
void indirect_projector_free(direct_projector * P){
	return;
}


#ifdef __cplusplus
}
#endif
