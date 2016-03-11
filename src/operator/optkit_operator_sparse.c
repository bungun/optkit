#include "optkit_operator_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

void * sparse_operator_data_alloc(sp_matrix * A)
{
	sparse_operator_data * op_data;
	op_data = malloc(sizeof(&op_data));
	blas_make_handle(&(op_data->dense_handle));
	sp_make_handle(&(op_data->sparse_handle));
	op_data->A = A;
	return (void *) op_data;
}

void sparse_operator_data_free(void * data)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_destroy_handle(op_data->sparse_handle);
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}


void sparse_operator_mul(void * data, vector * input, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasNoTrans, kOne, op_data->A,
		input, kZero, output);
}

void sparse_operator_mul_t(void * data, vector * input, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasTrans, kOne, op_data->A,
		input, kZero, output);
}

void sparse_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasNoTrans, alpha, op_data->A,
		input, beta, output);
}

void sparse_operator_mul_t_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	sparse_operator_data * op_data = (sparse_operator_data *) data;
	sp_blas_gemv(op_data->sparse_handle, CblasTrans, alpha, op_data->A,
		input, beta, output);
}


operator * sparse_operator_alloc(sp_matrix * A)
{
	OPTKIT_OPERATOR sparseop_kind = (A->order = CblasColMajor) ?
					SparseCSC_Operator :
					SparseCSR_Operator;
	return operator_alloc(
		sparseop_kind,
		A->size1, A->size2,
		sparse_operator_alloc(A),
		sparse_operator_mul,
		sparse_operator_mul_t,
		sparse_operator_mul_fused,
		sparse_operator_mul_t_fused,
		sparse_operator_data_free
	);
}

#ifdef __cplusplus
}
#endif