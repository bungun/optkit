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
	return operator_alloc(OkOperatorDiagonal, d->size, d->size,
		diagonal_operator_data_alloc(d),
		diagonal_operator_mul, diagonal_operator_mul_t,
		diagonal_operator_mul_fused, diagonal_operator_mul_t_fused,
		diagonal_operator_data_free
	);
}

static ok_status diagonal_operator_typecheck(operator * A, const char * caller)
{
	if (A->kind != OkOperatorDiagonal) {
		printf("diagonal_operator_%s() %s %s\n", caller, "undefined for",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	} else {
		return OPTKIT_SUCCESS;
	}
}

ok_status diagonal_operator_abs(operator * A)
{
	ok_status err = diagonal_operator_typecheck(A, "abs");
	diagonal_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (diagonal_operator_data *) A->data;
		vector_abs(op_data->d);
	}
	return err;
}

ok_status diagonal_operator_pow(operator * A, const ok_float power)
{
	ok_status err = diagonal_operator_typecheck(A, "pow");
	diagonal_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (diagonal_operator_data *) A->data;
		vector_pow(op_data->d, power);
	}
	return err;
}

ok_status diagonal_operator_scale(operator * A, const ok_float scaling)
{
	ok_status err = diagonal_operator_typecheck(A, "scale");
	diagonal_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (diagonal_operator_data *) A->data;
		vector_scale(op_data->d, scaling);
	}
	return err;
}

#ifdef __cplusplus
}
#endif