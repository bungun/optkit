#include "optkit_operator_dense.h"

#ifdef __cplusplus
extern "C" {
#endif


/* DENSE LINEAR OPERATOR */
void * dense_operator_data_alloc(matrix * A)
{
	dense_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
	blas_make_handle(&(op_data->dense_handle));
	op_data->A = A;
	return (void *) op_data;
}

void dense_operator_data_free(void * data)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}

void dense_operator_mul(void * data, vector * input, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasNoTrans, kOne, op_data->A, input,
		kZero, output);
}

void dense_operator_mul_t(void * data, vector * input, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasTrans, kOne, op_data->A, input,
		kZero, output);
}

void dense_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasNoTrans, alpha, op_data->A, input,
		beta, output);
}

void dense_operator_mul_t_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	blas_gemv(op_data->dense_handle, CblasTrans, alpha, op_data->A, input,
		beta, output);
}

operator * dense_operator_alloc(matrix * A)
{
	return operator_alloc(Dense_Operator, A->size1, A->size2,
		dense_operator_data_alloc(A),
		dense_operator_mul, dense_operator_mul_t,
		dense_operator_mul_fused, dense_operator_mul_t_fused,
		dense_operator_data_free
	);
}

#ifdef __cplusplus
}
#endif