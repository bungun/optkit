#include "optkit_operator_diagonal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DIAGONAL LINEAR OPERATOR */
void * diagonal_operator_data_alloc(vector * d)
{
	ok_status err = OPTKIT_SUCCESS;
	diagonal_operator_data * op_data = OK_NULL;

	if (!d || !d->data)
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	if (!err) {
		ok_alloc(op_data, sizeof(*op_data));
		op_data->d = d;
		err = OK_SCAN_ERR( blas_make_handle(&(op_data->dense_handle)) );
		if (err) {
			diagonal_operator_data_free(op_data);
			op_data = OK_NULL;
		}
	}
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

abstract_operator * diagonal_operator_alloc(vector * d)
{
	abstract_operator * o = OK_NULL;
	void * data;

	if (d && d->data) {
		data = diagonal_operator_data_alloc(d);
		if (data) {
			ok_alloc(o, sizeof(*o));
			o->kind = OkOperatorDiagonal;
			o->size1 = d->size;
			o->size2 = d->size;
			o->data = data;
			o->apply = diagonal_operator_mul;
			o->adjoint = diagonal_operator_mul_t;
			o->fused_apply = diagonal_operator_mul_fused;
			o->fused_adjoint = diagonal_operator_mul_t_fused;
			o->free = diagonal_operator_data_free;
		}
	}
	return o;
}

static ok_status diagonal_operator_typecheck(abstract_operator * A, const char * caller)
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

ok_status diagonal_operator_abs(abstract_operator * A)
{
	OK_RETURNIF_ERR( diagonal_operator_typecheck(A, "abs") );
	return vector_abs(((diagonal_operator_data *) A->data)->d);
}

ok_status diagonal_operator_pow(abstract_operator * A, const ok_float power)
{

	OK_RETURNIF_ERR( diagonal_operator_typecheck(A, "pow") );
	return vector_pow(((diagonal_operator_data *) A->data)->d, power);
}

ok_status diagonal_operator_scale(abstract_operator * A, const ok_float scaling)
{
	OK_RETURNIF_ERR( diagonal_operator_typecheck(A, "scale") );
	return vector_scale(((diagonal_operator_data *) A->data)->d, scaling);
}

#ifdef __cplusplus
}
#endif
