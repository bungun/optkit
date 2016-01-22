#include "optkit_operator.h"

#ifdef __cplusplus
extern "C" {
#endif


/* DENSE LINEAR OPERATOR */
void 
dense_operator_data_alloc(void ** data){
	dense_operator_data * op_data = (dense_operator_data *) malloc(
		sizeof(dense_operator_data) );
	blas_make_handle(&(op_data->dense_handle));
	*data = (void *) op_data;
}

void 
dense_operator_data_free(void * data){
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}

void 
dense_operator_mul(void * data, vector * input, vector * output){
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasNoTrans, 
		kOne, op_data->A, input, kZero, ouput);
}

void 
dense_operator_mul_t(void * data, vector * input, vector * output){
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasTrans, 
		kOne, op_data->A, input, kZero, ouput);
}

dense_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output){
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasNoTrans, 
		alpha, op_data->A, input, beta, ouput);
}

dense_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output){
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasTrans, 
		alpha, op_data->A, input, beta, ouput);
}

operator_t *
dense_operator_alloc(matrix * A){
	operator_t * op = (operator_t *) malloc( sizeof(operator_t));
	op->size1 = A->size1;
	op->size2 = A->size2;
	dense_operator_data_alloc(&(op->data));
	op->data->A = A;
	op->apply = dense_operator_mul;
	op->adjoint = dense_operator_mul_t;
	op->fused_apply = dense_operator_mul_fused;
	op->fused_adjoint = dense_operator_mul_t_fused;
	op->kind = Dense_Operator;
	return op;
}

/* SPARSE LINEAR OPERATOR */
void 
sparse_operator_data_alloc(void ** data){
	sparse_operator_data * op_data = (sparse_operator_data *) malloc(
		sizeof(sparse_operator_data) );
	blas_make_handle(&(op_data->dense_handle));
	sp_make_handle(&(op_data->sparse_handle));
	*data = (void *) op_data;
}

void 
sparse_operator_data_free(void * data){
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_destroy_handle(op_data->sparse_handle);
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}


void 
sparse_operator_mul(void * data, vector * input, vector * output){
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasNoTrans, 
		kOne, op_data->A, input, kZero, output);
}

void 
sparse_operator_mul_t(void * data, vector * input, vector * output){
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasTrans,
		kOne, op_data->A, input, kZero, output);
}

void 
sparse_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output){
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasNoTrans, 
		alpha, op_data->A, input, beta, output);
}

void 
sparse_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output){
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasTrans,
		alpha, op_data->A, input, beta, output);
}


operator_t *
sparse_operator_alloc(sp_matrix * A){
	operator_t * op = (operator_t *) malloc( sizeof(operator_t));
	op->size1 = A->size1;
	op->size2 = A->size2;
	sparse_operator_data_alloc(&(op->data));
	op->data->A = A;
	op->apply = sparse_operator_mul;
	op->adjoint = sparse_operator_mul_t;
	op->fused_apply = sparse_operator_mul_fused;
	op->fused_adjoint = sparse_operator_mul_t_fused;
	op->kind = (A->rowmajor == CblasRowMajor) SparseCSR_Operator : SparseCSC_Operator;
	return op;
}

/* GENERIC OPERATOR FREE */
void 
operator_free(operator_t * op){
	switch( op->kind ){
		case Dense_Operator : dense_operator_data_free(op->data);
		case Sparse_Operator_CSC : sparse_operator_data_free(op->data);
		case Sparse_Operator_CSR : sparse_operator_data_free(op->data);
		default : return;
	}
	ok_free(op);
}


#ifdef __cplusplus
}
#endif
