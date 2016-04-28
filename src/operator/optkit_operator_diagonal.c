#include "optkit_operator_diagonal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DIAGONAL LINEAR OPERATOR */
void * diagonal_operator_data_alloc(vector * d)
{
	diagonal_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
	memset(op_data, 0, sizeof(*op_data));
	blas_make_handle(&(op_data->dense_handle));
	op_data->d = d;
	return (void *) op_data;
}

ok_status diagonal_operator_data_free(void * data)
{
	diagonal_operator_data * op_data = (diagonal_operator_data *) data;
	OK_CHECK_PTR(op_data);
	ok_status err = blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
	return OK_SCAN_ERR( err );
}

ok_status diagonal_operator_mul(void * data, vector * input, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_diagmv(((diagonal_operator_data *) data)->dense_handle,
		kOne, ((diagonal_operator_data *) data)->d, input, kZero,
		output);
}

ok_status diagonal_operator_mul_t(void * data, vector * input, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_diagmv(((diagonal_operator_data *) data)->dense_handle,
		kOne, ((diagonal_operator_data *) data)->d, input, kZero,
		output);
}

ok_status diagonal_operator_mul_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_diagmv(((diagonal_operator_data *) data)->dense_handle,
		alpha, ((diagonal_operator_data *) data)->d, input, beta,
		output);
}

ok_status diagonal_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_diagmv(((diagonal_operator_data *) data)->dense_handle,
		alpha, ((diagonal_operator_data *) data)->d, input, beta,
		output);
}

operator * diagonal_operator_alloc(vector * d)
{
	operator * o = OK_NULL;
	if (d && d->data) {
		o = malloc(sizeof(*o));
		memset(o, 0, sizeof(*o));
		o->kind = OkOperatorDiagonal;
		o->size1 = d->size;
		o->size2 = d->size;
		o->data = diagonal_operator_data_alloc(d);
		o->apply = diagonal_operator_mul;
		o->adjoint = diagonal_operator_mul_t;
		o->fused_apply = diagonal_operator_mul_fused;
		o->fused_adjoint = diagonal_operator_mul_t_fused;
		o->free = diagonal_operator_data_free;
	}
	return o;
}

static ok_status diagonal_operator_typecheck(operator * A, const char * caller)
{
	OK_CHECK_OPERATOR(A);
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
	OK_RETURNIF_ERR( diagonal_operator_typecheck(A, "abs") );
	return vector_abs(((diagonal_operator_data *) A->data)->d);
}

ok_status diagonal_operator_pow(operator * A, const ok_float power)
{

	OK_RETURNIF_ERR( diagonal_operator_typecheck(A, "pow") );
	return vector_pow(((diagonal_operator_data *) A->data)->d, power);
}

ok_status diagonal_operator_scale(operator * A, const ok_float scaling)
{
	OK_RETURNIF_ERR( diagonal_operator_typecheck(A, "scale") );
	return vector_scale(((diagonal_operator_data *) A->data)->d, scaling);
}

#ifdef __cplusplus
}
#endif
