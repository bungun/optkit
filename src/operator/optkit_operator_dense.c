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
	if (operator->kind != OkOperatorDense) {
		printf("dense_operator_%s() %s %s\n", caller, "undefined for",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	} else {
		return OPTKIT_SUCCESS
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


ok_status dense_operator_copy(operator * A, void * data,
	OPTKIT_COPY_DIRECTION dir)
{
	dense_operator_data * op_data = OK_NULL;
	dense_operator_data * input_data = OK_NULL;
	dense_operator_caller_data * caller_data = OK_NULL;
	ok_status err = dense_operator_typecheck(A, "copy");

	if (!data) {
		printf("%s\n", "argument 'data' is null");
		err = OPTKIT_ERROR;
	}

	if (!err) {
		op_data = (dense_operator_data *) A->data;

	 	if (dir == OptkitToOptkit) {
			input_data = (dense_operator_data * data);
			sp_matrix_memcpy_mm(op_data->A, input_data->A);
		} else {
			caller_data = (dense_operator_caller_data *) data;
			if (!caller_data->val) {
				printf("%s%s%s\n", "argument 'data' ",
					"must contain valid pointer",
					"<ok_float *> 'val");
				err = OPTKIT_ERROR;
			} else if (caller_data->val != CblasRowMajor ||
				   caller_data->order != CblasColMajor) {
				printf("%s%s\n", "argument 'data' ",
					"must contain valid CblasOrder enum.");
				err = OPTKIT_ERROR;
			} else if (dir == OptkitToCaller) {
				matrix_memcpy_am(caller_data->val, op_data->A,
					caller_data->order);
			} else {
				matrix_memcpy_ma(op_data->A, caller_data->val,
					caller_data->order);
			}
		}
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