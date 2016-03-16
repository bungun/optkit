#include "optkit_operator_diagonal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DIAGONAL LINEAR OPERATOR */
void * diagonal_operator_data_alloc(vector * d)
{
	diagonal_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
	blas_make_handle(&(op_data->dense_handle));
	op_data->d = d;
	return (void *) op_data;
}

void diagonal_operator_data_free(void * data)
{
	diagonal_operator_data * op_data = (diagonal_operator_data *) data;
	blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
}

void diagonal_operator_mul(void * data, vector * input, vector * output)
{
	diagonal_operator_data * op_data = (diagonal_operator_data *) data;
	blas_diagmv(op_data->dense_handle, kOne, op_data->d, input, kZero,
		output);
}

void diagonal_operator_mul_t(void * data, vector * input, vector * output)
{
	diagonal_operator_data * op_data = (diagonal_operator_data *) data;
	blas_diagmv(op_data->dense_handle, kOne, op_data->d, input, kZero,
		output);
}

void diagonal_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	diagonal_operator_data * op_data = (diagonal_operator_data *) data;
	blas_diagmv(op_data->dense_handle, alpha, op_data->d, input, beta,
		output);
}

void diagonal_operator_mul_t_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	diagonal_operator_data * op_data = (diagonal_operator_data *) data;
	blas_diagmv(op_data->dense_handle, alpha, op_data->d, input, beta,
		output);
}

operator * diagonal_operator_alloc(vector * d)
{
	return operator_alloc(Diagonal_Operator, d->size, d->size,
		diagonal_operator_data_alloc(d),
		diagonal_operator_mul, diagonal_operator_mul_t,
		diagonal_operator_mul_fused, diagonal_operator_mul_t_fused,
		diagonal_operator_data_free
	);
}

#ifdef __cplusplus
}
#endif