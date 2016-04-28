#include "optkit_operator_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DENSE LINEAR OPERATOR */
void * dense_operator_data_alloc(matrix * A)
{
	dense_operator_data * op_data;
	op_data = malloc(sizeof(*op_data));
	memset(op_data, 0, sizeof(*op_data));
	blas_make_handle(&(op_data->dense_handle));
	op_data->A = A;
	return (void *) op_data;
}

ok_status dense_operator_data_free(void * data)
{
	dense_operator_data * op_data = (dense_operator_data *) data;
	OK_CHECK_PTR(op_data);
	ok_status err = blas_destroy_handle(op_data->dense_handle);
	ok_free(op_data);
	return OK_SCAN_ERR( err );
}

ok_status dense_operator_mul(void * data, vector * input, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_gemv(((dense_operator_data *) data)->dense_handle,
		CblasNoTrans, kOne, ((dense_operator_data *) data)->A, input,
		kZero, output);
}

ok_status dense_operator_mul_t(void * data, vector * input, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_gemv(((dense_operator_data *) data)->dense_handle,
		CblasTrans, kOne, ((dense_operator_data *) data)->A, input,
		kZero, output);
}

ok_status dense_operator_mul_fused(void * data, ok_float alpha, vector * input,
	ok_float beta, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_gemv(((dense_operator_data *) data)->dense_handle,
		CblasNoTrans, alpha, ((dense_operator_data *) data)->A, input,
		beta, output);
}

ok_status dense_operator_mul_t_fused(void * data, ok_float alpha,
	vector * input, ok_float beta, vector * output)
{
	OK_CHECK_PTR(data);
	return blas_gemv(((dense_operator_data *) data)->dense_handle,
		CblasTrans, alpha, ((dense_operator_data *) data)->A, input,
		beta, output);
}

operator * dense_operator_alloc(matrix * A)
{
	operator * o = OK_NULL;
	if (A && A->data) {
		o = malloc(sizeof(*o));
		memset(o, 0, sizeof(*o));
		o->kind = OkOperatorDense;
		o->size1 = A->size1;
		o->size2 = A->size2;
		o->data = dense_operator_data_alloc(A);
		o->apply = dense_operator_mul;
		o->adjoint = dense_operator_mul_t;
		o->fused_apply = dense_operator_mul_fused;
		o->fused_adjoint = dense_operator_mul_t_fused;
		o->free = dense_operator_data_free;
	}
	return o;
}

static ok_status dense_operator_typecheck(operator * A, const char * caller)
{
	OK_CHECK_OPERATOR(A);
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

	if (!err)
		return ((dense_operator_data *) A->data)->A;
	else
		return OK_NULL;
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

void * dense_operator_import(operator * A, void * data)
{
	ok_status err = dense_operator_typecheck(A, "import");
	dense_operator_data * op_data = OK_NULL;
	ok_float * import = OK_NULL;

	if (!err && data) {
		op_data = (dense_operator_data *) A->data;
		import = (ok_float *) data;
		matrix_memcpy_ma(op_data->A, import, op_data->A->order);
		ok_free(import);
		data = OK_NULL;
	}

	return data;
}

ok_status dense_operator_abs(operator * A)
{
	OK_RETURNIF_ERR( dense_operator_typecheck(A, "abs") );
	return matrix_abs(((dense_operator_data *) A->data)->A);
}

ok_status dense_operator_pow(operator * A, const ok_float power)
{
	OK_RETURNIF_ERR( dense_operator_typecheck(A, "pow") );
	return matrix_pow(((dense_operator_data *) A->data)->A, power);
}

ok_status dense_operator_scale(operator * A, const ok_float scaling)
{
	OK_RETURNIF_ERR( dense_operator_typecheck(A, "scale") );
	return matrix_scale(((dense_operator_data *) A->data)->A, scaling);
}

ok_status dense_operator_scale_left(operator * A, const vector * v)
{
	OK_RETURNIF_ERR( dense_operator_typecheck(A, "scale_left") );
	OK_CHECK_VECTOR(v);
	return matrix_scale_left(((dense_operator_data *) A->data)->A, v);
}

ok_status dense_operator_scale_right(operator * A, const vector * v)
{
	OK_RETURNIF_ERR( dense_operator_typecheck(A, "scale_right") );
	OK_CHECK_VECTOR(v);
	return matrix_scale_right(((dense_operator_data *) A->data)->A, v);
}

transformable_operator * dense_operator_to_transformable(operator * A)
{
	ok_status err = dense_operator_typecheck(A, "to_transformable");
	transformable_operator * t = OK_NULL;

	if (!err) {
		t = malloc(sizeof(*t));
		memset(t, 0, sizeof(*t));
		t->o = A;
		t->export = dense_operator_export;
		t->import = dense_operator_import;
		t->abs = dense_operator_abs;
		t->pow = dense_operator_pow;
		t->scale = dense_operator_scale;
		t->scale_left = dense_operator_scale_left;
		t->scale_right = dense_operator_scale_right;
	}
	return t;
}

#ifdef __cplusplus
}
#endif
