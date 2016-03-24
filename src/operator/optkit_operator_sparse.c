#include "optkit_operator_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

void * sparse_operator_data_alloc(sp_matrix * A)
{
	sparse_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
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
	OPTKIT_OPERATOR sparseop_kind = (A->order == CblasColMajor) ?
					OkOperatorSparseCSC :
					OkOperatorSparseCSR;

	return operator_alloc(sparseop_kind, A->size1, A->size2,
		sparse_operator_data_alloc(A),
		sparse_operator_mul, sparse_operator_mul_t,
		sparse_operator_mul_fused, sparse_operator_mul_t_fused,
		sparse_operator_data_free
	);
}

static ok_status sparse_operator_typecheck(operator * A,
	const char * caller)
{
	if (operator->kind != OkOperatorSparseCSR &&
	    operator->kind != OkOperatorSparseCSC) {
		printf("sparse_operator_%s() %s %s\n", caller, "undefined for",
			optkit_op2str(A->kind));
		return OPTKIT_ERROR;
	} else {
		return OPTKIT_SUCCESS
	}
}


sp_matrix * sparse_operator_get_matrix_pointer(operator * A)
{
	ok_status err = sparse_operator_typecheck(A, "get_matrix_pointer");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		return op_data->A;
	} else {
		return OK_NULL;
	}
}

ok_status sparse_operator_copy(operator * A, void * data,
	enum OPTKIT_COPY_DIRECTION dir)
{
	sparse_operator_data * op_data = OK_NULL;
	sparse_operator_data * input_data = OK_NULL;
	sparse_operator_caller_data * caller_data = OK_NULL;
	ok_status err = sparse_operator_typecheck(A, "copy");

	if (!data) {
		printf("%s\n", "argument 'data' is null");
		err = OPTKIT_ERROR;
	}

	if (!err) {
		op_data = (sparse_operator_data *) A->data;

	 	if (dir == OptkitToOptkit) {
			input_data = (sparse_operator_data * data);
			sp_matrix_memcpy_mm(op_data->A, input_data->A);
		} else {
			caller_data = (sparse_operator_caller_data *) data;
			if (!caller_data->val ||
			    !caller_data->ind ||
			    !caller_data->ptr) {
				printf("%s%s%s%s\n", "argument 'data' ",
					"must contain valid pointers",
					"<ok_float *> 'val', <ok_int *> 'ind',",
					" and <ok_int *> 'ptr'");
			err = OPTKIT_ERROR;
			} else if (dir == OptkitToCaller) {
				sp_matrix_memcpy_am(caller_data->val,
					caller_data->ind, caller_data->ptr,
					op_data->A);
			} else {
				sp_matrix_memcpy_ma(op_data->sparse_handle,
					op_data->A, caller_data->val,
					caller_data->ind, caller_data->ptr);
			}
		}
	}
	return err;
}

ok_status sparse_operator_abs(operator * A)
{
	ok_status err = sparse_operator_typecheck(A, "abs");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_abs(op_data->A);
	}
	return err;
}

ok_status sparse_operator_pow(operator * A, const ok_float power)
{
	ok_status err = sparse_operator_typecheck(A, "pow");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_pow(op_data->A, power);
	}
	return err;
}

ok_status sparse_operator_scale(operator * A, const ok_float scaling)
{
	ok_status err = sparse_operator_typecheck(A, "scale");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_scale(op_data->A, scaling);
	}
	return err;
}

ok_status sparse_operator_scale_left(operator * A, const vector * v)
{
	ok_status err = sparse_operator_typecheck(A, "scale_left");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_scale_left(op_data->A, v);
	}
	return err;
}

ok_status sparse_operator_scale_right(operator * A, const vector * v)
{
	ok_status err = sparse_operator_typecheck(A, "scale_right");
	sparse_operator_data * op_data = OK_NULL;

	if (!err) {
		op_data = (sparse_operator_data *) A->data;
		sp_matrix_scale_right(op_data->A, v);
	}
	return err;
}

#ifdef __cplusplus
}
#endif