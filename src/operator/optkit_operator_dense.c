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
	return operator_alloc(OkOperatorDense, A->size1, A->size2,
		dense_operator_data_alloc(A),
		dense_operator_mul, dense_operator_mul_t,
		dense_operator_mul_fused, dense_operator_mul_t_fused,
		dense_operator_data_free
	);
}

static ok_status dense_operator_typecheck(operator * A, const char * caller)
{
	if (A->kind != OkOperatorDense) {
		printf("dense_operator_%s() %s %s\n", caller, "undefined for",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	} else {
		return OPTKIT_SUCCESS;
	}
}

matrix * dense_operator_get_matrix_pointer(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "get_matrix_pointer");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		return op_data->A;
	} else {
		return OK_NULL;
	}
}


void * dense_operator_export(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "export");
	dense_operator_data * op_data = OK_NULL;
	ok_float * export = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		export = malloc(op_data->A->size1 * op_data->A->size2 *
			sizeof(*export));
		matrix_memcpy_am(export, op_data->A, op_data->A->order);
	}
	return (void *) export;
}

ok_status dense_operator_import(operator * A, void * data)
{
	ok_status err = dense_operator_typecheck(A, "import");
	dense_operator_data * op_data = OK_NULL;
	ok_float * import = OK_NULL;

	if (!err && data) {
		op_data = (dense_operator_data *) A->data;
		import = (ok_float *) data;
		matrix_memcpy_ma(op_data->A, import, op_data->A->order);
		ok_free(import);
	}

	return err;
}

ok_status dense_operator_abs(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "abs");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_abs(op_data->A);
	}
	return err;
}

ok_status dense_operator_pow(operator * A, const ok_float power)
{
	ok_status err = dense_operator_typecheck(A, "pow");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_pow(op_data->A, power);
	}
	return err;
}

ok_status dense_operator_scale(operator * A, const ok_float scaling)
{
	ok_status err = dense_operator_typecheck(A, "scale");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_scale(op_data->A, scaling);
	}
	return err;
}

ok_status dense_operator_scale_left(operator * A, const vector * v)
{
	ok_status err = dense_operator_typecheck(A, "scale_left");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_scale_left(op_data->A, v);
	}
	return err;
}

ok_status dense_operator_scale_right(operator * A, const vector * v)
{
	ok_status err = dense_operator_typecheck(A, "scale_right");
	dense_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (dense_operator_data *) A->data;
		matrix_scale_right(op_data->A, v);
	}
	return err;
}

#ifdef __cplusplus
}
#endif